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

from lm_eval import evaluator, tasks, utils
from lm_eval import evaluator, utils
from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from megatron_patch.model.qwen.gpt_model import GPTModel
from megatron_patch.arguments import get_patch_args
from megatron_patch.lm_evaluate import EvalHarnessAdaptor

def get_model_provider():
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()
        config = core_transformer_config_from_args(args)
        model = GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

        return model

    return model_provider

def main():

    args = get_args()
    initialize_tasks(args.verbosity)
    tasks_list = args.task_list.split(",")
    task_dict = tasks.get_task_dict(tasks_list)
    adaptor = EvalHarnessAdaptor(pretrained=args.load,
                                 batch_size=args.micro_batch_size,
                                 trust_remote_code=True,
                                 max_length=args.max_position_embeddings,
                                 model_provider=get_model_provider())

    results = evaluator.evaluate(adaptor, task_dict, limit=None)

    if results is not None:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_patch_args)
    main()
