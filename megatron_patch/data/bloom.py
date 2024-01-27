# Copyright (c) 2023 Alibaba PAI Team.
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

import json
import numpy as np
import torch

from megatron_patch.tokenizer import get_tokenizer

class BloomRawDataset(torch.utils.data.Dataset):
    """A class for processing a Bloom text dataset"""
    def __init__(self, datapaths, max_seq_length):
        """
        Initializes the dataset.
        Args:
            path(str): The path of the dataset file.
            tokenizer(object): The tokenizer object.
            max_seq_length(int): The maximum length of sequences.
        """
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        self.prompt = ''
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(
                self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample, self.tokenizer,
                                                   self.max_seq_length)

    def truncate(self, tokenizer, array, max_length):
        """
        Truncates an array to a maximum length or pads it with zeros if its length is less than `max_length`.
        Args:
            tokenizer: The tokenizer used to encode the input.
            array: The numpy array to truncate or pad.
            max_length: The maximum length of the array.
        Returns:
            A numpy array of length `max_length` containing the contents of `array`, truncated if necessary or padded with zeros.
        """

        if len(array) < max_length:
            return np.pad(array, (0, max_length - len(array)),
                          constant_values=tokenizer.eod)
        else:
            return array[:max_length]

    def process_samples_from_single_path(self, filename):
        """
        Process a single file containing prompt-answer pairs and return a list of samples.
        """

        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for example in f:
                text = json.loads(example)['text']
                sample = {
                    'prompt':
                    text + '</s>' if not text.endswith('</s>') else text,
                    'answer': text,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer,
                                       max_seq_length):
        """
        Convert a single sample containing a prompt-answer pair into a format suitable for GPT training.
        """

        tokens = tokenizer(sample['prompt'])
        input_ids = tokens['input_ids']
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length + 1)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample