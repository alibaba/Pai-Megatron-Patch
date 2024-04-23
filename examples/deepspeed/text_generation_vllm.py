#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from vllm import LLM, SamplingParams


def get_tasks_args(parser):
    group = parser.add_argument_group(title='vllm inference')
    group.add_argument('--checkpoint', type=str, help='The checkpoint of the the specified model', required=True)
    group.add_argument('--input-file', type=argparse.FileType('r'), help='Specify the input file', required=True)
    group.add_argument('--input-key', type=str, help='Specify the input key', default='query')
    group.add_argument('--output-file', type=argparse.FileType('w'), help='Specify the output file', required=True)
    group.add_argument('--batch-size', default=10, type=int, help='Specify the batch size of data')
    group.add_argument('--tensor-parallel-size', default=1, type=int, help='Specify the parallel size of tensor')
    group.add_argument('--output-max-tokens', default=2048, type=int, help='Specify the output max tokens')
    group.add_argument('--cuda-visible-devices', default='0,1,2,3,4,5,6,7', type=str, help='Specify the cuda visible devices')
    group.add_argument('--temperature', default=0, type=float, help='Specify the model temperature')
    return parser


parser = argparse.ArgumentParser(description='vllm Inference')
parser = get_tasks_args(parser)
args = parser.parse_args()
# 如果tp=1且指定多个，默认用第一个，例如：CUDA_VISIBLE_DEVICES=0,1,2,3，默认用0，tp>1的情况以此类推默认用前tp个；
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices


def batch_record(file, num=10):
    records = []
    for line in file:
        record = json.loads(line)
        # prompt = record['query']
        records.append(record)
        if len(records) >= num:
            _records = records
            records = []
            yield _records
    if len(records) > 0:
        yield records


def main():
    llm = LLM(model=args.checkpoint, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.output_max_tokens)

    ofile = args.output_file
    for records in batch_record(args.input_file, args.batch_size):
        inputs = [_x[args.input_key] for _x in records]
        outputs = llm.generate(inputs, sampling_params)
        for record, output in zip(records, outputs):
            generated_text = output.outputs[0].text
            if len(output.outputs[0].token_ids) < args.output_max_tokens:
                record["output"] = f"<s>{generated_text}</s>"
            else:
                record["output"] = f"<s>{generated_text}"
            json.dump(record, ofile, ensure_ascii=False)
            ofile.write('\n')
    ofile.close()


if __name__ == '__main__':
    main()
