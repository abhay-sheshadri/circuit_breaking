import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import concatenate_datasets, load_dataset

from .utils import *


class PaddingDataCollator:
    def __init__(self, pad_token_id):
        assert pad_token_id is not None, "pad_token_id must be specified"
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_size = len(batch)
        prompt_lengths = [len(x["input_ids"]) for x in batch]
        pad_length = max(prompt_lengths)

        tokens = torch.zeros(batch_size, pad_length, dtype=torch.long)
        mask = torch.zeros(batch_size, pad_length, dtype=torch.bool)

        for i, pl in enumerate(prompt_lengths):
            
            tokens[i, :pl] = torch.tensor(batch[i]["input_ids"], dtype=torch.long)
            tokens[i, pl:] = self.pad_token_id
            mask[i, :pl] = torch.tensor(batch[i]["mask"], dtype=torch.bool)

        return {
            "tokens": tokens,
            "mask": mask,
        }


def load_dataset_from_name(dataset_name, num_examples):
    # Load huggingface dataset by name
    if dataset_name == "pile_val":
        dataset = load_dataset("mit-han-lab/pile-val-backup")["validation"]
    elif dataset_name == "owt":
        dataset = load_dataset("Skylion007/openwebtext")["train"]
    else:
        dataset = load_dataset('json', data_files=f'datasets/{dataset_name}.jsonl')["train"]

    # Choose num_examples datapoints
    if num_examples is not None:
        if num_examples > len(dataset):
            num_examples = len(dataset)
        dataset = dataset.select(range(num_examples))

    return dataset


def create_dataset(
    dataset_name,
    tokenizer,
    max_seq_length=1024,
    num_examples=None,
    batch_size=32,
):

    # Check that the dataset is in the list of valid datasets
    list_of_dataset_names = ["owt", "pile_val", "harry_potter", "math_and_physics", "machine_learning"]
    assert dataset_name in list_of_dataset_names
    
    # Load the dataset and select num_examples
    dataset = load_dataset_from_name(dataset_name, num_examples)

    # Tokenize the entire dataset
    tokenizer.pad_token = tokenizer.eos_token    
    def tokenize_dataset(data):
        if dataset_name in ["pile_val", "owt"]:
            tokenized = tokenizer.batch_encode_plus(data["text"], truncation=True, max_length=max_seq_length)
            tokens = tokenized["input_ids"]
            mask = tokenized["attention_mask"]
        else:
            prompt_tokenized = tokenizer.batch_encode_plus(data["prompt"])
            completion_tokenized = tokenizer.batch_encode_plus(data["completion"], add_special_tokens=False)
            tokens = []
            mask = []
            for i in range(len(prompt_tokenized["input_ids"])):
                tokens_i = prompt_tokenized["input_ids"][i] + completion_tokenized["input_ids"][i]
                tokens.append(tokens_i[:max_seq_length])
                mask_i = [0,] * len(prompt_tokenized["attention_mask"][i]) + completion_tokenized["attention_mask"][i]
                mask.append(mask_i[:max_seq_length])
        return {
            "input_ids": tokens,
            "mask": mask
        }

    tokenized_dataset = dataset.map(
        tokenize_dataset,
        batched=True,
        batch_size=1000,
    )
    
    # Remove irrelevant columns
    all_columns = tokenized_dataset.column_names
    columns_to_remove = [column for column in all_columns
                            if column != 'input_ids' and column != 'mask' ]
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    # Use DataCollator to handle padding during training
    data_collator = PaddingDataCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    # Convert dataset to DataLoader for batch processing
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    return dataset, dataloader


def filter_and_rewrite_dataset(dataset_name, model, output_path):
    dataset, dataloader = create_dataset(dataset_name, model.tokenizer, 2048, None,)
    results = compute_metrics(model, dataloader)
    
    dataset = dataset.add_column("loss", results["loss"])
    ds_sorted = dataset.sort("loss")

    num_examples = len(ds_sorted)
    top_75_percent_index = int(0.75 * num_examples)
    bottom_75_percent = ds_sorted.select(range(top_75_percent_index))
    
    bottom_75_percent = bottom_75_percent.remove_columns(["loss"])
    bottom_75_percent.to_json(output_path, orient="records", lines=True)
