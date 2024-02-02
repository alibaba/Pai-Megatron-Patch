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

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.generation.gpt_predictor import GPTPredictor
from megatron_patch.model.qwen.gpt_model import GPTModel
from megatron_patch.arguments import get_patch_args

class MegatronGPTPredictor(GPTPredictor):
    def model_provider(self, pre_process=True, post_process=True):
        args = get_args()
        build_tokenizer(args)
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            parallel_output = False
        else:
            parallel_output = True
        config = core_transformer_config_from_args(get_args())
        model = GPTModel(config,
                         num_tokentypes=0,
                         parallel_output=parallel_output,
                         pre_process=pre_process,
                         post_process=post_process)
        return model


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    predictor = MegatronGPTPredictor()
    predictor.predict()
