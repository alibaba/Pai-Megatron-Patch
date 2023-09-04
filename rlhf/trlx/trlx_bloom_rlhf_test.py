import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List

import json
import torch
from reward_model.reward_model_bloom import BLOOMRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

REWARD_CHECKPOINT_PATH = "bigscience/bloom-7b1"
SFT_MODEL_PATH = "bigscience/bloom-7b1"

config = TRLConfig(
    train=TrainConfig(
        seq_length=550,
        epochs=20,
        total_steps=100000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=2000,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path="bigscience/bloom-7b1",
        num_layers_unfrozen=8,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="bigscience/bloom-7b1",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 50,
        },
    ),
)


def read_json(data_path):
    res = []
    with open(data_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            line = json.loads(line)
            res.append(line)
    return res


def create_prompt_dataset(path):
    dataset = read_json(path)
    print("dataset_size: ", len(dataset))
    print("dataset_size case: ", dataset[0])
    data_list = []
    for sample in tqdm(dataset):
        data_dict = {}
        data_dict["prompt"] = sample["query"]
        data_dict["label"] = sample["reference"]
        data_list.append(data_dict)
    print("data_nums: ", len(data_list))
    return data_list


if __name__ == "__main__":
    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_tokenizer.padding_side = 'right'
    rw_model = BLOOMRewardModel("bigscience/bloom-7b1")
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [chosen + "</s>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i],
                    truncation=True,
                    max_length=max_length,  # [canceled] to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):
        original_samples = [text for text in samples]
        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    train_dataset = create_prompt_dataset('./rl_data/train_data.json')
    val_dataset = create_prompt_dataset('./rl_data/val_data.json')

    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in train_dataset]
    val_set = [(sample["prompt"], sample["label"]) for sample in val_dataset]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
