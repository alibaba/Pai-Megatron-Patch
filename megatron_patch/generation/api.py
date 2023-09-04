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

import torch

from megatron.core import mpu
from megatron.text_generation.communication import broadcast_float_list
from .generation import beam_search_and_return_on_first_stage
from .generation import generate_tokens_probs_and_return_on_first_stage
from .generation import score_and_return_on_first_stage
from .tokenization import detokenize_generations, tokenize_prompts


def generate_and_post_process(model,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1):
    """
    Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list.

    Args:
        model (torch.nn.Module): The model used for text generation.
        prompts (List[str], optional): A list of prompts to generate text from.
        tokens_to_generate (int): The maximum number of tokens to generate.
        return_output_log_probs (bool): A flag indicating whether to return the output log probabilities for each generated token.
        top_k_sampling (int): The value of k for top-k sampling.
        top_p_sampling (float): The value of p for top-p sampling.
        top_p_decay (float): The amount by which to decay the value of p for each token generated.
        top_p_bound (float): The minimum value of p for top-p sampling.
        temperature (float): The temperature value to apply during sampling.
        add_BOS (bool): A flag indicating whether to add a beginning-of-sentence token to the generated output.
        use_eod_token_for_early_termination (bool): A flag indicating whether to use the end-of-document token for early termination.
        stop_on_double_eol (bool): A flag indicating whether to stop generating text when a double end-of-line token is generated.
        stop_on_eol (bool): A flag indicating whether to stop generating text when an end-of-line token is generated.
        prevent_newline_after_colon (bool): A flag indicating whether to prevent newline characters after a colon.
        random_seed (int): The random seed to use for text generation.

    Returns:
        Tuple[List[str], List[str], List[List[float]], List[int]]: A tuple containing the following elements:
            - prompts_plus_generations (List[str]): A list of prompts followed by the generated text.
            - prompts_plus_generations_segments (List[str]): A list of segments corresponding to each prompt and generated text.
            - output_log_probs (List[List[float]]): The output log probabilities for each generated token (if return_output_log_probs is True).
            - tokens (List[int]): The generated tokens.

    """

    # Main inference.
    tokens, lengths, output_log_probs = generate(
        model,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=return_output_log_probs,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        tokens, prompts_plus_generations, prompts_plus_generations_segments = \
            detokenize_generations(tokens, lengths, True)

        if return_output_log_probs:
            output_log_probs = output_log_probs.cpu().numpy().tolist()
            for i, (prob, seg) in enumerate(
                    zip(output_log_probs, prompts_plus_generations_segments)):
                output_log_probs[i] = prob[:len(seg) - 1]

        return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens

    return None


def generate(model,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             top_p_decay=0.0,
             top_p_bound=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             prevent_newline_after_colon=False,
             random_seed=-1):
    """
    Given prompts and input parameters, run inference and return the generated tokens,
    lengths, and output log probabilities.

    Args:
        model (torch.nn.Module): The model used for text generation.
        prompts (List[str], optional): A list of prompts to generate text from.
        tokens_to_generate (int): The maximum number of tokens to generate.
        return_output_log_probs (bool): A flag indicating whether to return the output log probabilities for each generated token.
        top_k_sampling (int): The value of k for top-k sampling.
        top_p_sampling (float): The value of p for top-p sampling.
        top_p_decay (float): The amount by which to decay the value of p for each token generated.
        top_p_bound (float): The minimum value of p for top-p sampling.
        temperature (float): The temperature value to apply during sampling.
        add_BOS (bool): A flag indicating whether to add a beginning-of-sentence token to the generated output.
        use_eod_token_for_early_termination (bool): A flag indicating whether to use the end-of-document token for early termination.
        stop_on_double_eol (bool): A flag indicating whether to stop generating text when a double end-of-line token is generated.
        stop_on_eol (bool): A flag indicating whether to stop generating text when an end-of-line token is generated.
        prevent_newline_after_colon (bool): A flag indicating whether to prevent newline characters after a colon.
        random_seed (int): The random seed to use for text generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following elements:
            - tokens (torch.Tensor): The prompt plus generated tokens.
            - lengths (torch.Tensor): The lengths of the prompt plus the generated tokens.
            - output_log_probs (torch.Tensor): The output log probabilities for each generated token.
    """

    # Make sure input params are avaialble to all ranks.
    values = [
        tokens_to_generate, return_output_log_probs, top_k_sampling,
        top_p_sampling, top_p_decay, top_p_bound, temperature, add_BOS,
        use_eod_token_for_early_termination, stop_on_double_eol, stop_on_eol,
        prevent_newline_after_colon, random_seed
    ]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        add_BOS=add_BOS)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(model, context_tokens_tensor,
                                               context_length_tensor)

    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage(
        model,
        context_tokens_tensor,
        context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon)


def beam_search_and_post_process(model,
                                 prompts=None,
                                 tokens_to_generate=0,
                                 beam_size=0,
                                 add_BOS=False,
                                 stop_token=50256,
                                 num_return_gen=1,
                                 length_penalty=1,
                                 prevent_newline_after_colon=False):
    """
    Run beam search and post-process outputs, i.e., detokenize,
    move to cpu and convert to list.
    Args:
        model (torch.nn.Module): The model used for beam search.
        prompts (List[List[int]], optional): List of prompts.
        tokens_to_generate (int, optional): Number of tokens to generate.
        beam_size (int, optional): Beam size for beam search.
        add_BOS (bool, optional): Whether to add the BOS token to the prompt.
        stop_token (int, optional): Token that indicates the end of generation.
        num_return_gen (int, optional): Number of generated sequences to return.
        length_penalty (float, optional): Length penalty for beam search.
        prevent_newline_after_colon (bool, optional): Whether to prevent newline after a colon. Defaults to False.
    Returns:
        Tuple[List[List[int]], List[List[int]], List[float]]: A tuple containing
        the post-processed generations, generation segments, and scores.
    """

    # Main inference.
    tokens, scores = beam_search(
        model,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        beam_size=beam_size,
        add_BOS=add_BOS,
        stop_token=stop_token,
        num_return_gen=num_return_gen,
        length_penalty=length_penalty,
        prevent_newline_after_colon=prevent_newline_after_colon)
    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        lengths = tokens.size(1) * torch.ones(
            beam_size, dtype=torch.int64, device=torch.cuda.current_device())
        tokens, prompts_plus_generations, prompts_plus_generations_segments = detokenize_generations(
            tokens, lengths, True)
        scores = scores.cpu().numpy().tolist()
        return prompts_plus_generations, prompts_plus_generations_segments, scores

    return None


def beam_search(model,
                prompts=None,
                tokens_to_generate=0,
                beam_size=0,
                add_BOS=False,
                stop_token=50256,
                num_return_gen=1,
                length_penalty=1,
                prevent_newline_after_colon=False):
    """
    Perform beam search to generate sequences.

    Args:
        model (torch.nn.Module): The model used for beam search.
        prompts (List[List[int]], optional): List of prompts, where each prompt is a list of token ids.
        tokens_to_generate (int, optional): Number of tokens to generate.
        beam_size (int, optional): Beam size for beam search.
        add_BOS (bool, optional): Whether to add the BOS token to the prompt.
        stop_token (int, optional): Token that indicates the end of generation.
        num_return_gen (int, optional): Number of generated sequences to return.
        length_penalty (float, optional): Length penalty for beam search.
        prevent_newline_after_colon (bool, optional): Whether to prevent newline.

    Returns:
        torch.Tensor: The generated tokens.
    """
    # Make sure input params are avaialble to all ranks.
    values = [
        tokens_to_generate, beam_size, add_BOS, stop_token, num_return_gen,
        length_penalty, prevent_newline_after_colon
    ]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    beam_size = int(values_float_tensor[1].item())
    add_BOS = bool(values_float_tensor[2].item())
    stop_token = int(values_float_tensor[3].item())
    num_return_gen = int(values_float_tensor[4].item())
    length_penalty = values_float_tensor[5].item()
    prevent_newline_after_colon = values_float_tensor[6].item()

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        add_BOS=add_BOS)

    return beam_search_and_return_on_first_stage(
        model,
        context_tokens_tensor,
        context_length_tensor,
        beam_size,
        stop_token=stop_token,
        num_return_gen=num_return_gen,
        length_penalty=length_penalty,
        prevent_newline_after_colon=prevent_newline_after_colon)
