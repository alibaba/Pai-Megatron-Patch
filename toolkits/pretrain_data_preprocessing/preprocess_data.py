# Copyright (c) 2023 Alibaba PAI Team.  All rights reserved.
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
"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import time
from threading import Semaphore
import torch
import ftfy
import lm_dataformat as lmd
import tqdm

from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            try:
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad',max_length=32768,truncation=True)['input_ids']
                """
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='max_length',
                                             max_length=2047, truncation=True)['input_ids']
                """
                if max(text_ids) >= Encoder.tokenizer.vocab_size:
                    print(text)
                    print(max(text_ids))
                    continue
            except Exception as e:
                print(f"Error encoding text: {e}")  # print error message
                continue
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                if hasattr(Encoder.tokenizer, 'eos_token_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eos_token_id)
                elif hasattr(Encoder.tokenizer, 'eod_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eod_id)
                else:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
                #doc_ids[-1].append(Encoder.tokenizer.pad_token_id)
            ids[key] = doc_ids
        return ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True)
    group.add_argument(
        '--jsonl-keys',
        nargs='+',
        default=['content'],
        help='space separate listed of keys to extract from jsonl. Defa',
    )
    group.add_argument(
        '--num-docs',
        default=None,
        type=int,
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=[
            'JiebaBPETokenizer', 'BloomTokenizerFromHF',
            'ChatGLMTokenizerFromHF', 'GPT2BPETokenizer',
            'GLM10BZHTokenizerFromHF', 'IcetkGLM130BTokenizer',
            'LLamaTokenizer', 'FalconTokenizer', 'OPTTokenizer',
            'StarcoderTokenizerFromHF', 'QwenTokenizer','Qwen2Tokenizer', 'MistralTokenizer'
        ],
        help='What type of tokenizer to use.',
    )
    group.add_argument('--vocab-file',
                       type=str,
                       default=None,
                       help='Path to the vocab file')

    group.add_argument(
        '--merge-file',
        type=str,
        default=None,
        help='Path to the BPE merge file (if necessary).',
    )
    group.add_argument(
        '--append-eod',
        action='store_true',
        help='Append an <eod> token to the end of a document.',
    )
    group.add_argument('--ftfy',
                       action='store_true',
                       help='Use ftfy to clean text')
    group = parser.add_argument_group(title='output data')
    group.add_argument(
        '--output-prefix',
        type=str,
        required=True,
        help='Path to binary output file without suffix',
    )
    group.add_argument(
        '--dataset-impl',
        type=str,
        default='mmap',
        choices=['lazy', 'cached', 'mmap'],
        help='Dataset implementation to use. Default: mmap',
    )

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    group.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Interval between progress updates',
    )
    group.add_argument('--load',
                       type=str,
                       default=None,
                       help='path to tokenizer config file')
    group.add_argument('--seq-length',
                       type=int,
                       default=2048,
                       help='sequence length')
    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='extra_vocab_size')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore):
    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    args.tensor_model_parallel_size = 1
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.vocab_extra_ids = 0
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f'Vocab size: {tokenizer.vocab_size}')
    print(f'Output prefix: {args.output_prefix}')

    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    file_list = os.listdir(args.input)
    path_list = [os.path.join(args.input, file) for file in file_list]
    fin = yield_from_files(path_list, semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers,
                                    initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = '{}_{}_{}.bin'.format(args.output_prefix, key,
                                                      'document')
        output_idx_files[key] = '{}_{}_{}.idx'.format(args.output_prefix, key,
                                                      'document')
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(f'Processed {i} documents '
                                 f' ({i / elapsed} docs/s, {mbs} MB/s).')
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()