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

import copy
import io
import json
import os
import random
import re
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

class AbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""
    def __init__(self, data_dir, data_name, file_name, tokenizer,
                 max_seq_length):
        """
        Initializes the dataset.
        Args:
            data_dir (str): The directory containing the dataset files.
            data_name (str): The name of the dataset.
            file_name (str): The name of the dataset file.
            tokenizer (Tokenizer): The tokenizer to use for encoding the dataset.
            max_seq_length (int): The maximum sequence length for the input.
        """
        # Store inputs.
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset_name = data_name
        self.samples = self.process_samples_from_single_path(
            os.path.join(data_dir, data_name, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        ids, types, paddings = self.build_tokens_types_paddings_from_text(
            raw_sample['text_a'], raw_sample['text_b'], self.tokenizer,
            self.max_seq_length)

        sample = self.build_sample(ids, types, paddings, raw_sample['label'],
                                   raw_sample['uid'])
        return sample

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass

    def build_tokens_types_paddings_from_text(self, text_a, text_b, tokenizer,
                                              max_seq_length):
        """Build token types and paddings,
        trim if needed, and pad if needed."""
        text_a_ids = tokenizer.tokenize(text_a)
        text_b_ids = None
        if text_b is not None:
            text_b_ids = tokenizer.tokenize(text_b)

        return self.build_tokens_types_paddings_from_ids(
            text_a_ids, text_b_ids, max_seq_length, tokenizer.cls,
            tokenizer.sep, tokenizer.pad)

    def build_tokens_types_paddings_from_ids(self, text_a_ids, text_b_ids,
                                             max_seq_length, cls_id, sep_id,
                                             pad_id):
        """
        Builds the token types and paddings based on the input text ids,
        and trims and pads the sequences if necessary.
        Args:
            text_a_ids (list[int]): The token ids of the input text A.
            text_b_ids (list[int]): The token ids of the input text B, or None if there is no text B.
            max_seq_length (int): The maximum sequence length.
            cls_id (int): The id of the [CLS] token.
            sep_id (int): The id of the [SEP] token.
            pad_id (int): The id of the padding token.
        Returns:
            tuple: The token ids, token types, and token paddings.
        """

        ids = []
        types = []
        paddings = []

        # [CLS].
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)

        # A.
        len_text_a = len(text_a_ids)
        ids.extend(text_a_ids)
        types.extend([0] * len_text_a)
        paddings.extend([1] * len_text_a)

        # [SEP].
        ids.append(sep_id)
        types.append(0)
        paddings.append(1)

        # B.
        if text_b_ids is not None:
            len_text_b = len(text_b_ids)
            ids.extend(text_b_ids)
            types.extend([1] * len_text_b)
            paddings.extend([1] * len_text_b)

        # Cap the size.
        trimmed = False
        if len(ids) >= max_seq_length:
            max_seq_length_m1 = max_seq_length - 1
            ids = ids[0:max_seq_length_m1]
            types = types[0:max_seq_length_m1]
            paddings = paddings[0:max_seq_length_m1]
            trimmed = True

        # [SEP].
        if (text_b_ids is not None) or trimmed:
            ids.append(sep_id)
            if text_b_ids is None:
                types.append(0)
            else:
                types.append(1)
            paddings.append(1)

        # Padding.
        padding_length = max_seq_length - len(ids)
        if padding_length > 0:
            ids.extend([pad_id] * padding_length)
            types.extend([pad_id] * padding_length)
            paddings.extend([0] * padding_length)

        return ids, types, paddings

    def build_sample(self, ids, types, paddings, label, unique_id):
        """
        Converts the token ids, types, paddings, label, and unique ID to a NumPy array and
        returns a sample to be consumed by the batch producer.
        Args:
            ids (list[int]): The token ids.
            types (list[int]): The token types.
            paddings (list[int]): The paddings.
            label (int): The label.
            unique_id (int): The unique ID.
        Returns:
            dict: The sample dictionary containing the token ids, types, paddings, label, and unique ID.
        """

        ids_np = np.array(ids, dtype=np.int64)
        types_np = np.array(types, dtype=np.int64)
        paddings_np = np.array(paddings, dtype=np.int64)
        sample = ({
            'text': ids_np,
            'types': types_np,
            'padding_mask': paddings_np,
            'label': int(label),
            'uid': int(unique_id)
        })

        return sample

    def clean_text(self, text):
        """
        Cleans the text by removing newlines and multiple spaces, and adjusting the end of sentence dot.
        Args:
            text (str): The text to be cleaned.
        Returns:
            str: The cleaned text.
        """

        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        for _ in range(3):
            text = text.replace(' . ', '. ')

        return text

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


class GPTDataset(AbstractDataset):
    """GPT dataset class."""
    def __init__(self, datapaths, tokenizer, max_seq_length):
        """
        Initializes a new instance of the GPTDataset class.
        Args:
            datapaths (list): List of file paths containing the raw text data.
            tokenizer: Instance of the tokenizer used to tokenize the text data.
            max_seq_length (int): Maximum sequence length for input to the model.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

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

    def clean_text(self, raw):
        """
        Cleans the input text by removing URLs, extra spaces, and special characters, and adjusting the end of sentence dot.
        Args:
            text (str): The raw text to be processed.
        Returns:
            str: The cleaned text.
        """

        httpcom = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|['
                             r'!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        raw = httpcom.sub('', raw)

        space = re.compile(r' +')
        raw = space.sub(' ', raw)

        fil = re.compile(
            u'[^0-9a-zA-Z\u4e00-\u9fa5.， ,\\-。'
            u'%《*》/•、&＆(—)（+）：？!！“”·]+', re.UNICODE)
        raw = fil.sub('', raw)
        return raw.strip()

    def process_samples_from_single_path(self, filename):
        """
        Process a single file and return a list of samples.
        """
        print(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        with open(filename, encoding='utf-8-sig') as f:
            for line in f:
                row = line.strip()
                sample = {
                    'text': row,
                }
                total += 1
                samples.append(sample)

        print(' >> processed {} samples.'.format(len(samples)))
        random.shuffle(samples)
        return samples

    def gpt_convert_example_to_feature(self, sample, tokenizer,
                                       max_seq_length):
        """
        Convert a single sample into a format suitable for GPT training.
        """
        tokens = tokenizer.tokenize(sample['text'])
        input_ids = np.array(tokens)
        input_ids = self.truncate(tokenizer, input_ids, max_seq_length)
        train_sample = {'input_ids': np.array(input_ids)}
        return train_sample