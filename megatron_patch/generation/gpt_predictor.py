# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import json
import torch

from megatron.core.enums import ModelType
from megatron import get_args
from megatron import get_timers
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint

from megatron_patch.generation.api import generate_and_post_process
from megatron_patch.tokenizer import build_tokenizer

class GPTPredictor():
    """A Predictor for model."""
    def __init__(self):
        super().__init__()

    def predict(self):
        """Run predict process """

        args = get_args()
        build_tokenizer(args)
        timers = get_timers()

        args.train_iters = 1
        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
        model = get_model(self.model_provider,
                          model_type=ModelType.encoder_or_decoder,
                          wrap_with_ddp=False)
        assert args.load is not None
        if args.load is not None and args.no_load_optim:
            load_checkpoint(model, None, None)
        timers('model-and-optimizer-setup').stop()
        torch.distributed.barrier()

        timers = get_timers()
        timers('load-checkpoint', log_level=0).start(barrier=True)
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
        timers.log(['model-and-optimizer-setup'])

        if not isinstance(model, list):
            model = [model]

        assert len(model) == 1, 'Above condition should have caught this'
        model = model[0]
        if args.text_generate_input_file != '':
            num_examples = len(open(args.text_generate_input_file).readlines())
            prompts = []
            pred_outputs = []
            with open(args.text_generate_input_file,
                      encoding='utf-8') as reader,\
                    open(args.text_generate_output_file,
                         'w', encoding='utf-8') as writer:
                buffer = []

                for idx, line in enumerate(reader):
                    line = line.strip()
                    json_obj = json.loads(line)
                    line = json_obj['query'][:args.seq_length]
                    prompts.append(line)
                    if len(buffer) < args.micro_batch_size:
                        buffer.append(line)

                    if len(
                            buffer
                    ) == args.micro_batch_size or idx == num_examples - 1:
                        sl = args.out_seq_length
                        tk = args.top_k
                        tp = args.top_p
                        temperature = args.temperature
                        prompts_plus_generations, _, _, _ = \
                            generate_and_post_process(model,
                                                      prompts=buffer,
                                                      tokens_to_generate=sl,
                                                      top_k_sampling=tk,
                                                      temperature=temperature,
                                                      top_p_sampling=tp)

                        for prompt, p_and_g in zip(buffer,
                                                   prompts_plus_generations):
                            generation = p_and_g.replace('<|endoftext|>', '')
                            print(p_and_g)
                            writer.write(generation + '\n')
                            pred_outputs.append(generation)
                        buffer.clear()

                    if idx % args.micro_batch_size == 0:
                        print('processed {} examples'.format(idx))
