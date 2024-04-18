# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
import numpy as np
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from megatron import get_args
from megatron.checkpointing import load_checkpoint
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.pipeline_parallel.p2p_communication import recv_forward
from megatron.core.pipeline_parallel.p2p_communication import send_forward
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.arguments import core_transformer_config_from_args

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM, eval_logger

from megatron_patch.training import get_model
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer

class EvalHarnessAdaptor(HFLM):

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        max_length: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.args = get_args()
        build_tokenizer(self.args)
        self.tokenizer = get_tokenizer()
        self.is_main = torch.distributed.get_rank() == 0
        self.adaptive_seq_len = self.args.adaptive_seq_len
        self.model_provider = kwargs['model_provider']

        super().__init__(pretrained=pretrained,
                         batch_size=batch_size,
                         trust_remote_code=trust_remote_code,
                         max_length=max_length,
                         tokenizer=self.tokenizer)

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        model_list = get_model(self.model_provider,
                          model_type=ModelType.encoder_or_decoder,
                          wrap_with_ddp=False)

        if pretrained is not None:
            load_checkpoint(model_list, None, None)

        self._model = model_list[0]

        def tie_weights(self):
            pass
        self._model.tie_weights = types.MethodType(tie_weights, self._model)

        return None

    def create_model_inputs(self, tokens):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.eot_token_id,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss)

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps, attn_mask=None, labels=None):
        args = get_args()

        # Since the shape of the micro-batch will change
        # We need set the correct shapes here
        # So that latter pipeline stages knows which shapes to expect.
        # Otherwise we will deadlock.

        args.micro_batch_size = len(inps)
        args.seq_length = len(inps[0])
        config = core_transformer_config_from_args(args)
        tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
        input_tensor = recv_forward(tensor_shape, config)

        # Forward pass through the model.
        unwrapped_model = unwrap_model(self.model)
        unwrapped_model.set_input_tensor(input_tensor)
        output = self.model(*self.create_model_inputs(inps)[0])
        send_forward(output, config)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)
                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)
