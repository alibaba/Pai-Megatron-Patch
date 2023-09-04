import json
import os

import torch
from torch.utils.data import Dataset

from reward_model_bloom import BLOOMRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def read_json(data_path):
    res = []
    with open(data_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            line = json.loads(line)
            res.append(line)
    return res


def create_comparison_dataset(path):
    dataset = read_json(path)
    print('dataset_size: ', len(dataset))
    print('dataset_size case: ', dataset[0])
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample['prompt']
        chosen_answer = sample['choosen']
        rejected_answer = sample['rejected']
        if chosen_answer == rejected_answer:
            continue
        if len(chosen_answer) < 1 or len(rejected_answer) < 1:
            continue
        pair['choosen'] = prompt + '\n' + chosen_answer
        pair['rejected'] = prompt + '\n' + rejected_answer
        pairs.append(pair)
    print('pairs_nums: ', len(pairs))
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair['choosen'], pair['rejected']
            # print("chosen: ", chosen)
            # print("rejected: ", rejected)
            chosen_encodings_dict = tokenizer(
                # "<|startoftext|>" + chosen + "<|endoftext|>",
                chosen + '</s>',
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            rejected_encodings_dict = tokenizer(
                # "<|startoftext|>" + rejected + "<|endoftext|>",
                rejected + '</s>',
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            # print("chosen_input_ids_shape: ", chosen_encodings_dict["input_ids"].size())
            # print("chosen_input_ids: ", chosen_encodings_dict["input_ids"])
            # print("rejected_input_ids: ", rejected_encodings_dict["input_ids"])
            # chosen_ids = chosen_encodings_dict["input_ids"]
            # rejected_ids = rejected_encodings_dict["input_ids"]
            # print("dengyu: ", (chosen_ids == rejected_ids))
            # print("equal: ", torch.eq(chosen_ids, rejected_ids))
            # print("all: ", torch.all(torch.eq(chosen_ids, rejected_ids)))

            if torch.all(
                    torch.eq(chosen_encodings_dict['input_ids'],
                             rejected_encodings_dict['input_ids'])).item():
                # print("chosen_input: ", tokenizer.decode(chosen_encodings_dict["input_ids"][0]))
                # print("rejected_input: ", tokenizer.decode(rejected_encodings_dict["input_ids"][0]))
                # print("chosen_input_ids: ", chosen_encodings_dict["input_ids"])
                # print("rejected_input_ids: ", rejected_encodings_dict["input_ids"])
                pass
            else:
                self.chosen_input_ids.append(
                    chosen_encodings_dict['input_ids'])
                self.chosen_attn_masks.append(
                    chosen_encodings_dict['attention_mask'])
                self.rejected_input_ids.append(
                    rejected_encodings_dict['input_ids'])
                self.rejected_attn_masks.append(
                    rejected_encodings_dict['attention_mask'])

        print('chosen_input_size: ', len(self.chosen_input_ids))
        print('rejected_input_size: ', len(self.rejected_input_ids))

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        # tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
        batch = {}
        batch['input_ids'] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data])
        # print("????input_ids: ", batch["input_ids"])
        # print("????input: ", tokenizer.decode(batch["input_ids"][0]))
        batch['attention_mask'] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data])
        batch['labels'] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(
        chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result['accuracy'] = acc

    return result


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b1')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!tokenizer.pad_token: ", tokenizer.pad_token)

    if not os.path.exists('rm_checkpoint'):
        os.mkdir('rm_checkpoint')

    training_args = TrainingArguments(
        output_dir='rm_checkpoint/',
        num_train_epochs=2,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy='steps',
        evaluation_strategy='steps',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=1000,
        save_steps=1000,
        warmup_steps=100,
        logging_dir='./logs',
        fp16=False,
        bf16=True,
        learning_rate=1e-5,
        deepspeed='ds_config_bloom.json',
        save_total_limit=3,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J

    model = BLOOMRewardModel('bigscience/bloom-1b1')

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = './ranking_data/'
    train_pairs = create_comparison_dataset(
        os.path.join(data_path, 'ranking_train.json'))
    val_pairs = create_comparison_dataset(
        os.path.join(data_path, 'ranking_val.json'))

    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs,
                                    tokenizer,
                                    max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
