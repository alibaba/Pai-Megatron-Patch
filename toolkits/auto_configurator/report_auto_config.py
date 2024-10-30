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

from typing import List, Optional
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.core.num_microbatches_calculator import get_num_microbatches
import megatron.core.num_microbatches_calculator as mb_calculator
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer

from report_theoretical_memory import report_theoretical_memory

modelsize_to_minworldsize_map = {
    "nparams-0.5B", "ws-8",
    "nparams-1.5B", "ws-8",
    "nparams-3B", "ws-8",
    "nparams-7B", "ws-8",
    "nparams-8B", "ws-8",
    "nparams-13B", "ws-16",
    "nparams-14B", "ws-16",
    "nparams-32B", "ws-16",
    "nparams-70B", "ws-32",
    "nparams-72B", "ws-32",
}

modelsize_minworldsize_to_maxmbs_maxseqlen_map = {
    ("nparams-0.5B", "ws-8"): [("mbs-4", "seqlen-4096"), ("mbs-2", "seqlen-8192")],
    ("nparams-1.5B", "ws-8"): [("mbs-4", "seqlen-4096"), ("mbs-2", "seqlen-8192")],
    ("nparams-3B", "ws-8"): [("mbs-2", "seqlen-4096"), ("mbs-2", "seqlen-8192")],
    ("nparams-7B", "ws-8"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-8B", "ws-8"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-13B", "ws-16"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-14B", "ws-16"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-32B", "ws-16"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-70B", "ws-32"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
    ("nparams-72B", "ws-32"): [("mbs-1", "seqlen-4096"), ("mbs-1", "seqlen-8192")],
}

modelsize_minworldsize_maxmbs_maxseqlen_to_bestconfig_map = {
    ("nparams-0.5B", "ws-8", "mbs-4", "seqlen-4096"): ("tp-1", "pp-1", "vp-1", "gbs-768"),
    ("nparams-0.5B", "ws-8", "mbs-2", "seqlen-8192"): ("tp-1", "pp-1", "vp-1", "gbs-768"),
    ("nparams-1.5B", "ws-8", "mbs-4", "seqlen-4096"): ("tp-1", "pp-1", "vp-1", "gbs-768"),
    ("nparams-1.5B", "ws-8", "mbs-2", "seqlen-8192"): ("tp-1", "pp-1", "vp-1", "gbs-768"),
    ("nparams-3B", "ws-8", "mbs-2", "seqlen-4096"): ("tp-1", "pp-1", "vp-1", "gbs-768"),
    ("nparams-3B", "ws-8", "mbs-2", "seqlen-8192"): ("tp-2", "pp-1", "vp-1", "gbs-768"),
    ("nparams-7B", "ws-8", "mbs-1", "seqlen-4096"): ("tp-1", "pp-1", "vp-1", "gbs-1024"),
    ("nparams-7B", "ws-8", "mbs-1", "seqlen-8192"): ("tp-2", "pp-1", "vp-1", "gbs-1024"),
    ("nparams-8B", "ws-8", "mbs-1", "seqlen-4096"): ("tp-1", "pp-1", "vp-1", "gbs-1024"),
    ("nparams-8B", "ws-8", "mbs-1", "seqlen-8192"): ("tp-2", "pp-1", "vp-1", "gbs-1024"),
    ("nparams-13B", "ws-16", "mbs-1", "seqlen-4096"): ("tp-2", "pp-1", "vp-1", "gbs-1536"),
    ("nparams-13B", "ws-16", "mbs-1", "seqlen-8192"): ("tp-4", "pp-1", "vp-1", "gbs-1536"),
    ("nparams-14B", "ws-16", "mbs-1", "seqlen-4096"): ("tp-2", "pp-1", "vp-1", "gbs-1536"),
    ("nparams-14B", "ws-16", "mbs-1", "seqlen-8192"): ("tp-4", "pp-1", "vp-1", "gbs-1536"),
    ("nparams-32B", "ws-16", "mbs-1", "seqlen-4096"): ("tp-4", "pp-2", "vp-2", "gbs-1536"),
    ("nparams-32B", "ws-16", "mbs-1", "seqlen-8192"): ("tp-8", "pp-2", "vp-2", "gbs-1536"),
    ("nparams-70B", "ws-32", "mbs-1", "seqlen-4096"): ("tp-4", "pp-4", "vp-4", "gbs-2304"),
    ("nparams-70B", "ws-32", "mbs-1", "seqlen-8192"): ("tp-8", "pp-4", "vp-4", "gbs-2304"),
    ("nparams-72B", "ws-32", "mbs-1", "seqlen-4096"): ("tp-4", "pp-4", "vp-4", "gbs-2304"),
    ("nparams-72B", "ws-32", "mbs-1", "seqlen-8192"): ("tp-8", "pp-4", "vp-4", "gbs-2304"),
}



def add_extra_args(parser):
    parser = get_patch_args(parser)

    def add_args(parser):
        parser.add_argument(
            "--world-size",
            type=int,
            default=1
        )

        parser.add_argument(
            "--model-size",
            type=str
        )

        return parser

    parser = add_args(parser)

    return parser

def report_auto_config(args, verbose=False):

    nparams = "nparams-"+args.model_size
    ws = "ws-"+str(args.world_size)
    seqlen = "seqlen-" + str(args.seq_length)

    if (nparams, ws) not in modelsize_minworldsize_to_maxmbs_maxseqlen_map:
        if nparams not in modelsize_to_minworldsize_map:
            raise ValueError("model size should be in [0.5B, 1.5B, 3B, 7B, 8B, 13B, 14B, 70B, 72B]")
        else:
            ws = modelsize_to_minworldsize_map[nparams]
            raise ValueError("you only need to set minimal world size {} for model size {}".format(ws, nparams))

    else:
        maxmbs_maxseqlen_list = modelsize_minworldsize_to_maxmbs_maxseqlen_map[(nparams, ws)]
        matched_max_mbs = None
        matched_max_seqlen = None
        condidate_seqlens = []
        for maxmbs, maxseqlen in maxmbs_maxseqlen_list:
            if seqlen == maxseqlen:
                matched_max_mbs = maxmbs
                matched_max_seqlen = maxseqlen
            condidate_seqlens.append(maxseqlen)

        if matched_max_mbs and matched_max_seqlen:

            best_config = modelsize_minworldsize_maxmbs_maxseqlen_to_bestconfig_map[(nparams, ws, matched_max_mbs, matched_max_seqlen)]
            tp_size = best_config[0].replace("tp-", "")
            pp_size = best_config[1].replace("pp-", "")
            vp_size = best_config[2].replace("vp-", "")
            gbs = best_config[3].replace("gbs-", "")

            args.micro_batch_size = int(maxmbs.replace("mbs-", ""))
            args.global_batch_size = int(gbs)
            args.tensor_model_parallel_size = int(tp_size)
            args.pipeline_model_parallel_size  = int(pp_size)
            if args.rank == 0:
                print(
                    f"--tensor-model-parallel-size={tp_size} \n"
                    f"--pipeline-model-parallel-size={pp_size} \n"
                    f"--num-layers-per-virtual-pipeline-stage={vp_size} \n"
                    f"--micro-batch-size={maxmbs} \n"
                    f"--global-batch-size={gbs} \n", flush=True
                )
        else:
            raise ValueError("the max seqlen for world size{} should be in {}".format(ws, condidate_seqlens))

def reconfigure_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
):
    """Reconfigure number of micro-batches calculator.

    Args:
        rank (int): Rank of the GPU, only rank 0 will log the information.
        rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        global_batch_size (int): Global batch size for the model.
        micro_batch_size (int): Micro batch size at initialization.
        data_parallel_size (int): Data parallel size.
    """

    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = mb_calculator.build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )

if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True, extra_args_provider=add_extra_args)
    args = get_args()
    build_tokenizer(args)
    report_auto_config(args, verbose=True)
    """
    args.world_size = args.num_nodes * args.num_gpus_per_node
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    data_parallel_size = args.world_size // (model_parallel_size * args.context_parallel_size)
    reconfigure_num_microbatches_calculator(0, None, args.global_batch_size, args.micro_batch_size, data_parallel_size)
    num_microbatches = get_num_microbatches()
    report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
    """