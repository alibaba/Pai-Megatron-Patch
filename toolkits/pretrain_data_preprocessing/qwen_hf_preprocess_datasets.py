import multiprocessing
import fire
import glob
import transformers
from datasets import load_dataset


def build_dataset(
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
        cache_dir: str,
):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    num_workers = 16

    raw_datasets = load_dataset(
        "json",
        data_files=data_path,
        split="train",
        cache_dir=cache_dir,
    )

    dataset = raw_datasets.map(
        lambda example: tokenizer(example["text"]),
        batched=True,
        batch_size=3000,
        num_proc=num_workers,
        remove_columns=raw_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenization",
    )

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts with sequence length {sequence_length}",
    )

    return dataset

def run_preprocess(jsonl_path, encoded_path, tokenizer, sequence_length, cache_dir):
    dataset = build_dataset(
        data_path=jsonl_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(encoded_path)

def main(
        data_dir: str,
        tokenizer_name_or_path: str,
        sequence_length: int = 2048,
        cache_dir: str = "./hf-cache",
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir,
        model_max_length=sequence_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="<|extra_0|>"))

    ds_paths = {}
    for file in glob.glob(data_dir+"/*.jsonl"):
        ds_paths[file] = file.replace(".jsonl", ".encode")

    for k, v in ds_paths.items():
        print("===========================================")
        print(k)
        print(v)
    print("===========================================")

    po = multiprocessing.Pool(8)
    for jsonl_path, encoded_path in ds_paths.items():
        po.apply_async(func=run_preprocess, args=(jsonl_path, encoded_path, tokenizer, sequence_length, cache_dir))
    po.close()
    po.join()
    print('done')


if __name__ == "__main__":
    fire.Fire(main)
