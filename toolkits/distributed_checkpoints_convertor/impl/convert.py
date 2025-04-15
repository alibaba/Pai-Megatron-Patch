import os
import torch
import time

from importlib import import_module
from functools import partial

# NOTE: Some models (e.g., moonlight) adopts a customed tokenizer, which 
# requires trust_remote_code=True
from megatron.training import global_vars
global_vars.build_tokenizer = partial(global_vars.build_tokenizer, trust_remote_code=True)
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

@torch.inference_mode()
def convert(
    synchronizer,
    pretrain_script,
    load_dir,
    save_dir,
    hf_dir: str = None,
    mcore2hf: bool = False
):
    sync_module = import_module(synchronizer)
    model_provider_func = None
    if pretrain_script is not None:
        pretrain_module = import_module(pretrain_script)
        model_provider_func = pretrain_module.model_provider
    if mcore2hf:
        args = get_args()
        args.load = load_dir
        synchronizer = sync_module.MG2HFSynchronizer(
            hf_dir, 
            model_provider_func=model_provider_func
        )
    else:
        synchronizer = sync_module.HF2MGSynchronizer(
            load_dir, 
            model_provider_func=model_provider_func
        )

    synchronizer.sync_params()
    synchronizer.check_and_save(save_dir)

def add_args(parser):
    group = parser.add_argument_group(title='Distributed CKPT Convertor')

    group.add_argument('--synchronizer', type=str, default='general', 
                       help='The module required for conversion, should have two class `MG2HFSynchronizer` and `HF2MGSynchronizer`')
    group.add_argument('--pretrain-script', type=str, default=None, 
                       help='The path of pretrain script, the model is built with the `model_provider()` in it, use `pretrain_gpt.py` by default.')
    group.add_argument('--model-type', type=str, required=True, choices=['GPT'])
    group.add_argument('--load-dir', type=str, required=True, help='Directory to load model checkpoint from')
    group.add_argument('--save-dir', type=str, required=True, help='Directory to save model checkpoint to')
    group.add_argument('--hf-dir', type=str, help='pretrained huggingface checkpoint directory')

    group.add_argument('--padded-vocab-size', type=int, default=None)
    group.add_argument('--target-ckpt-format', default='torch_dist', choices=['torch_dist'], help='Checkpoint format to use.')
    group.add_argument('--use-gpu', action='store_true')
    group.add_argument('--mcore2hf', action='store_true')
    group.add_argument('--debug', action='store_true', help='enable debug mode to check the integrity of conversion, could be slow for large models')
    group.add_argument('--dryrun', action='store_true', help='If set, the conversion will be performed on an empty model and no save occurs, only for mcore2hf debugging.')
    group.add_argument('--num-hf-saver', type=int, default=None, help='Set the amount of huggingface savers in mcore2hf mode, each saver will requires extra memorys for comm, set smaller if OOM occurs, by default it is world_size.')
    group.add_argument('--max-shard-size', type=str, default='4GB', help='Set the sharded size, reduce memory consumption if smaller')
    
    return parser

if __name__ == '__main__':
    start_time = time.time()
    initialize_megatron(extra_args_provider=add_args, allow_no_cuda=True)
    args = get_args()
    actual_world_size = int(os.environ["WORLD_SIZE"])
    actual_rank = int(os.environ["RANK"])
    if args.num_hf_saver is None:
        args.num_hf_saver = actual_world_size
    print(f"WORLD_SIZE: {actual_world_size}, RANK: {actual_rank}, LOCAL_RANK: {args.local_rank}")

    if args.target_ckpt_format == 'torch_dist':
        convert(
            args.synchronizer,
            args.pretrain_script,
            args.load_dir,
            args.save_dir,
            hf_dir=args.hf_dir,
            mcore2hf=args.mcore2hf
        )
    else:
        raise NotImplementedError()

    torch.distributed.barrier()
    end_time = time.time()
    if actual_rank == 0:
        print(f"Conversion finished in {end_time - start_time} seconds.")