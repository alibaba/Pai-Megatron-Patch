#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import json
import time
import datetime
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_tasks_args(parser):
    group = parser.add_argument_group(title='vllm inference')
    group.add_argument('--checkpoint', type=str, help='The checkpoint of the the specified model', required=True)
    group.add_argument('--input-file', type=argparse.FileType('r'), help='Specify the input file', required=True)
    group.add_argument('--output-file', type=argparse.FileType('w'), help='Specify the output file', required=True)
    group.add_argument('--cuda-visible-devices', type=int, default=0, help='Cuda visible devices', required=False)
    group.add_argument('--output-max-tokens', default=2048, type=int, help='Specify the output max tokens')
    return parser


parser = argparse.ArgumentParser(description='Deepspeed Inference')
parser = get_tasks_args(parser)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, device_map="auto", trust_remote_code=True)

    st = time.time()
    token_len_list = []
    with open(args.output_file, 'w') as ofile:
        for line in open(args.input_file):
            record = json.loads(line)
            start_time = datetime.datetime.now()
            inputs = tokenizer.encode(record['query'], return_tensors="pt").to(model.device)
            outputs = model.generate(inputs, max_new_tokens=args.output_max_tokens)
            token_length = len(outputs[0]) - len(inputs[0])
            token_len_list.append(token_length)
            print(datetime.datetime.now() - start_time, 'token_length:', token_length)
            ofile.write(tokenizer.decode(outputs[0]) + '\n')

    total_time = time.time() - st
    print(args.output_file, 'total_time: {}, avg: {} token/s'.format(total_time, sum(token_len_list) / total_time))


if __name__ == '__main__':
    main()
