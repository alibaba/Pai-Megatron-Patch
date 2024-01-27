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

import argparse
import logging
import sys
import copy
import os
os.environ["WANDB_DISABLED"] = "true"
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    Trainer,
)
from datasets import load_dataset

def get_patch_args(parser):

    group = parser.add_argument_group(title='starcoder')

    group.add_argument('--local-rank', type=int, default=None,
                        help='local rank passed from distributed launcher')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')

    group.add_argument('--gradient-accumulation-steps', type=int, default=None)

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')

    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')

    group.add_argument('--intermediate-size',
                       type=int,
                       default=None,
                       help='--intermediate-size')

    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')

    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')

    group.add_argument('--num-workers',
                       type=int,
                       default=None)

    group.add_argument('--logging-dir',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--train-data',
                       default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')

    group.add_argument('--valid-data',
                       default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--enable-gradient-checkpointing', action='store_true')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')

    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')

    group.add_argument('--load', type=str, default=None)

    group.add_argument('--save', type=str, default=None)

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')


    return parser

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='PyTorch Starcoder Training')
parser = get_patch_args(parser)
args = parser.parse_args()


def tokenize(strings, tokenizer):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='np',
            padding='max_length',
            max_length=args.seq_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        tokenize(strings, tokenizer)
        for strings in (examples, sources)
    ]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels,
                                 sources_tokenized['input_ids_lens']):
        label[:source_len] = -100
    return dict(input_ids=input_ids, labels=labels)

def main():
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed='./ds_config.json',
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        do_eval=False,
        per_device_eval_batch_size=args.micro_batch_size,
        evaluation_strategy="epoch",
        dataloader_num_workers=args.num_workers,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        logging_strategy='steps',
        log_level='info',
        logging_dir=args.logging_dir,
        logging_steps=1,
        disable_tqdm=False,
        ddp_timeout=18000
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    transformers.set_seed(training_args.seed)

    data_files = {}
    if os.path.isdir(args.train_data):
        file_names = [os.path.join(args.train_data, f) for f in os.listdir(args.train_data)]
        data_files["train"] = file_names
    else:
        data_files["train"] = args.train_data

    if os.path.isdir(args.valid_data):
        file_names = [os.path.join(args.valid_data, f) for f in os.listdir(args.valid_data)]
        data_files["validation"] = file_names
    else:
        data_files["validation"] = args.valid_data

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=None,
        use_auth_token=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.load,
        use_fast=True,
        revision='main',
        use_auth_token=None,
        max_length=args.seq_length,
        padding='max_length',
        truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        args.load,
        from_tf=False,
        revision='main',
        use_auth_token=None,
        low_cpu_mem_usage=False,
    )
    model.to('cuda')
    if args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_params = model.num_parameters()
    logger.info(f"Training model with {n_params * 1e-9:.2f}B model")
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        sources = examples['instruction']
        targets = examples['output']
        data_dict = preprocess(sources, targets, tokenizer)
        return data_dict

    with training_args.main_process_first(desc="dataset map tokenization"):
        lm_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=64
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
