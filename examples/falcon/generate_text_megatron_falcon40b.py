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
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron

from megatron_patch.generation.gpt_predictor import GPTPredictor
from megatron_patch.model.falcon40b.gpt_model import GPTModel
from megatron_patch.arguments import get_patch_args

class MegatronGPTPredictor(GPTPredictor):
    def model_provider(self, pre_process=True, post_process=True):
        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        model = GPTModel(num_tokentypes=0,
                         parallel_output=False,
                         pre_process=pre_process,
                         post_process=post_process)

        return model


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    predictor = MegatronGPTPredictor()
    predictor.predict()
