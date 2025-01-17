import os
import numpy as np
import torch

from collections.abc import Mapping, Sequence
@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

def build_layer_id_mapping(args):
    """
        global layer id <--> local layer id
    """
    ltog, gtol = dict(), dict()

    assert args.target_decoder_first_pipeline_num_layers is None or args.target_num_layers_per_virtual_pipeline_stage is None, "Currently uneven VPP not supported"

    if args.target_decoder_first_pipeline_num_layers is not None:
        remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size - 1
        assert remained_layers % remained_stages == 0
        pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)

        for pp_id, num_layers in enumerate(pp_layers_per_stage):
            for global_layer_id in range(offset, offset + num_layers):
                # NOTE: map a global transformer layer to a local pp rank
                # global_id <--> (pp_id, vpp_id, local_id)
                local_layer_id = global_layer_id - offset
                ltog[(pp_id, 0, local_layer_id)] = global_layer_id
                gtol[global_layer_id] = (pp_id, 0, local_layer_id)
            offset += num_layers
    else:
        n_chunks = args.pipeline_model_parallel_size
        pp_size = args.pipeline_model_parallel_size
        if args.target_num_layers_per_virtual_pipeline_stage is not None:
            assert args.num_layers % (args.target_num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size) == 0
            n_chunks = args.num_layers // args.target_num_layers_per_virtual_pipeline_stage
        num_layer_per_chunk = args.num_layers // n_chunks
        pp_layers_per_stage = [num_layer_per_chunk] * n_chunks

        offset = 0        
        for chunk_id, num_layers in enumerate(pp_layers_per_stage):
            for global_layer_id in range(offset, offset + num_layers):
                # NOTE: map a global transformer layer to a local pp rank
                # global_id <--> (pp_id, vpp_id, local_id)
                pp_id = chunk_id % pp_size
                vpp_id = chunk_id // pp_size
                local_layer_id = global_layer_id - offset
                ltog[(pp_id, vpp_id, local_layer_id)] = global_layer_id
                gtol[global_layer_id] = (pp_id, vpp_id, local_layer_id)
            offset += num_layers
    return ltog, gtol

def safe_copy(src_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    assert src_tensor.dtype == dst_tensor.dtype
    assert src_tensor.shape == dst_tensor.shape
    dst_tensor.data.copy_(src_tensor.data)
    return src_tensor.numel()

def save_state_dict(args, model_chunks, checkpoint_name, has_vpp: bool=False):
    """
    Save some model chunks to a megatron checkpoint file
    """
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0    
    if not has_vpp:
        state_dict['model'] = model_chunks[0]
    else:
        for vpp_id in range(len(model_chunks)):
            state_dict[f"model{vpp_id}"] = model_chunks[vpp_id]
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)
