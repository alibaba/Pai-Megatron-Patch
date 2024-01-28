#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import argparse
from vllm import LLM, SamplingParams


def get_patch_args(parser):
    group = parser.add_argument_group(title='vllm inference')
    group.add_argument('--checkpoint', type=str, help='The checkpoint of the the specified model', required=True)
    group.add_argument('--input-file', type=argparse.FileType('r'), help='Specify the input file', required=True)
    group.add_argument('--output-file', type=argparse.FileType('w'), help='Specify the output file', required=True)
    group.add_argument('--output-max-tokens', default=2048, type=int, help='Specify the output file')
    return parser


parser = argparse.ArgumentParser(description='vllm Inference')
parser = get_patch_args(parser)
args = parser.parse_args()


def batch_prompt(file, num=10):
    prompts = []
    for line in file:
        record = json.loads(line)
        prompt = record['query']
        prompts.append(prompt)
        if len(prompts) >= num:
            _prompts = prompts
            prompts = []
            yield _prompts
    if len(prompts) > 0:
        yield prompts


def main():
    llm = LLM(model=args.checkpoint, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_max_tokens)

    ofile = args.output_file
    for prompts in batch_prompt(args.input_file):
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            if len(output.outputs[0].token_ids) < args.output_max_tokens:
                ofile.write(f"<s>{generated_text}</s>\n")
            else:
                ofile.write(f"<s>{generated_text}\n")
    ofile.close()


if __name__ == '__main__':
    main()
