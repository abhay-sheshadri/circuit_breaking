import argparse
import random

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_gpu(model):
    # Move object to cpu and deallocate VRAM
    model.cpu()
    torch.cuda.empty_cache()


def load_model_from_transformers(model_name_or_path):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    return model, tokenizer


def from_hf_to_tlens(hf_model, hf_tokenizer, model_name, disable_grads=False):
    # Convert huggingface model to transformer lens
    clear_gpu(hf_model)
    hooked_model = HookedTransformer.from_pretrained_no_processing(
        model_name, hf_model=hf_model, tokenizer=hf_tokenizer, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hooked_model.cuda()
    if disable_grads:
        for param in hooked_model.parameters():
             param.requires_grad_(False)
    return hooked_model

 
def compute_metrics(model, dataloader, verbose=True):
    # Compute 
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=model.tokenizer.pad_token_id
    )  

    results = {
        "num_tokens": [],
        "loss": [],
        "top1_acc": [],
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss", disable=not verbose):
            # When the dataset is chunked, the leftover piece is kept. 
            # However, sometimes the leftover piece is of size 1, and should be 
            # skipped.
            if batch["tokens"].shape[1] <= 1:
                continue

            # Forward pass through the model
            input_ids = batch["tokens"][:, :-1].to("cuda")
            targets = batch["tokens"][:, 1:].to("cuda")
            mask = batch["mask"][:, 1:].to("cuda")
            logits = model(input_ids)

            # Mask out prompt tokens
            mask = rearrange(mask, "b n -> (b n)")
            logits = rearrange(logits, "b n c -> (b n) c")
            logits = logits[mask]
            targets = rearrange(targets, "b n -> (b n)")
            targets = targets[mask]
            results["num_tokens"].append(targets.shape[0])

            # Add cross entropy loss to the results
            loss = criterion(logits, targets)
            results["loss"].append(loss.item())

            # Add top1 accuracy to the results
            acc = (logits.argmax(-1) == targets).to(torch.float32).mean()
            results["top1_acc"].append(acc.item())

    # Calculate summary of results
    loss_array = np.array(results["loss"])
    acc_array = np.array(results["top1_acc"])
    results["summary"] = {
        "loss_mean": np.mean(loss_array[~np.isnan(loss_array)]),
        "loss_var": np.var(loss_array[~np.isnan(loss_array)]),
        "top1_acc": np.mean(acc_array[~np.isnan(acc_array)]),
        "top1_var": np.var(acc_array[~np.isnan(acc_array)]),
    }
    
    return results

    
def to_boolean_top(original_dict, prune=0.01):
    # Flatten all tensors and concatenate them into one big tensor to find the top 1% value
    all_values = torch.cat([tensor.flatten() for tensor in original_dict.values()])
    # Find the threshold for the top 1%
    kth_value = int( (1-prune) * all_values.numel())
    threshold = all_values.kthvalue(kth_value).values
    # Create a new dictionary where tensors have boolean values indicating top 1% values
    boolean_dict = {}
    for key, tensor in original_dict.items():
        boolean_dict[key] = tensor > threshold
    return boolean_dict


def lor_dicts(dict1, dict2):
    # Create a new dict where every element is the logical or of the two input dicts
    new_dict = {}
    for key in dict1:
        new_dict[key] = torch.logical_or(dict1[key], dict2[key])
    return new_dict


def get_deep_attribute(obj, attr_string):
    attributes = attr_string.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj