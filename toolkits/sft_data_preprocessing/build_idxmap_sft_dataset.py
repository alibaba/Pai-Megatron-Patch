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

import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import multiprocessing

from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.seq_length = self.args.seq_length

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)


    def encode_blocked(self, datas):
        return list(self.encode(datas))

    def encode(self, datas):
        if isinstance(datas, dict):
            datas = [datas]
        
        ids = {}
        lens = {}
        doc_ids = []
        sentence_lens = []
        label_ids = []

        pad_token_id = self.tokenizer.pad_token_id
        # NOTE: in SFT, any tokenizer is required to:
        # (1) have a conversation chat_template
        # (2) the generated assistant input_ids are after the system/user input_ids
        # With (2), input_mask will be genarated

        # WARNING: the seqlen of built idxmap dataset is 2x of input seqlen!!!!
        for data in datas:
            text = [
                {'role': 'user', 'content': data["instruction"]+data['input']},
                {'role': 'assistant', 'content': data['output']}
            ]
            input_ids = self.tokenizer.apply_chat_template(text[:-1])
            if len(input_ids) >= self.seq_length:
                print('Extreme long user input, omitted...')
                continue
            all_ids = self.tokenizer.apply_chat_template(text)
            if len(all_ids) >= self.seq_length:
                print('Extreme long sequence, truncted...')
                all_ids = all_ids[:self.seq_length]

            for t1, t2 in zip(input_ids, all_ids):
                assert t1 == t2, "The user input_ids are not a prefix of the full input_ids!"

            y_ids = [-100] * (len(input_ids) - 1) + all_ids[len(input_ids):] + [-100]
            all_ids[-1] = - 1 - all_ids[-1]

            if sum(sentence_lens) + len(all_ids) > self.seq_length:
                if self.seq_length > sum(sentence_lens):
                    doc_ids = doc_ids + [pad_token_id] * (self.seq_length - sum(sentence_lens))
                    label_ids = label_ids + [-100] * (self.seq_length - sum(sentence_lens))     
                ids['text'] = doc_ids + label_ids
                lens['text'] = [len(doc_ids) * 2]
                yield ids, lens, len(json.dumps(ids))               
                ids = {}
                lens = {}
                doc_ids = []
                sentence_lens = []
                label_ids = []
            
            doc_ids.extend(all_ids)
            label_ids.extend(y_ids)
            sentence_lens.append(len(all_ids))

        if sum(sentence_lens) > 0:
            # Need Padding
            if self.seq_length > sum(sentence_lens):
                doc_ids = doc_ids + [pad_token_id] * (self.seq_length - sum(sentence_lens))
                label_ids = label_ids + [-100] * (self.seq_length - sum(sentence_lens))
            ids['text'] = doc_ids + label_ids
            lens['text'] = [len(doc_ids) * 2]
        yield ids, lens, len(json.dumps(ids))
    
    
class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers
        self.args = get_args()

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)

        # json or jsonl
        try:
            with open(input_file_name, 'r', encoding='utf-8') as f:
                fin = json.load(f)
        except Exception:
            fin = []
            with open(input_file_name, 'r', encoding='utf-8') as f:
                fin = [json.loads(d) for d in f.readlines()]
        
        assert isinstance(fin, list)
        # NOTE: each item in fin is a group (dict / list[dict]) of samples may be packed together
    
        startup_start = time.time()
        encoder = Encoder(self.args)
        if self.args.sequence_packing:
            # collect
            tmp = []
            for d in fin:
                if isinstance(d, dict):
                    tmp.append(d)
                else:
                    tmp.extend(d)
            fin = tmp
            encoder.initializer()
            # NOTE: single thread for packing
            print(f"Raw Dataset has {len(fin)} samples")
            encoded_docs = (encoder.encode(fin),)
        else:
            if self.args.debug:
                encoder.initializer()
                encoded_docs = (encoder.encode_blocked(doc) for doc in fin)
            else:
                pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
                encoded_docs = pool.imap(encoder.encode_blocked, fin, 32)

        tokenizer = build_tokenizer(self.args)
        level = "document"
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        cnt = 1
        for datas in encoded_docs:
            for (doc, sentence_lens, bytes_processed) in datas:
                total_bytes_processed += bytes_processed
                for key in doc.keys():
                    builders[key].add_document(doc[key], sentence_lens[key])
                self.print_processing_stats(cnt, proc_start, total_bytes_processed)
                cnt += 1
        print(f"After pre-tokenizing, the idxmap dataset has {cnt - 1} samples")

        builders[key].finalize(output_idx_files[key])

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False, default='GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--sequence-packing',action='store_true', help='packing sequence')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--debug', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=['Qwen2Tokenizer', 'LLamaTokenizer', 'DeepSeekV2Tokenizer', 'LLama3Tokenizer'],
        help='What type of tokenizer to use.',
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
                       default=0,
                       help='extra_vocab_size')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")
    
    if args.sequence_packing:
        print('Use internal single-threaded sequence packing..')
    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        file_list = os.listdir(args.input)
        in_file_names = [os.path.join(args.input, file) for file in file_list]

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1)%args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    # encode partition files in parallel
    processes = []
    input_key = 'partition'

    for name in in_ss_out_names:
        if args.debug:
            partition.process_json_file((name[input_key], name['output_prefix']))
        else:

            p = multiprocessing.Process(target=partition.process_json_file,
                                        args=((name[input_key], name['output_prefix']),))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':

    main()