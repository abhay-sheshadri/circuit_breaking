import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from itertools import cycle
from transformer_lens import HookedTransformer
from .masks import BasicMask
from tqdm import tqdm


def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner
    """
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all
    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0
    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()
    return loss


def get_minibatch(batch, start_idx, length):
    if batch is None:
        return None
    # Divide the batch into smaller batches that can fit on the GPU
    new_batch = {}
    for key in batch:
        new_batch[key] = batch[key][start_idx:start_idx+length]
    return new_batch


def zero_nan_grads(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"Parameter {name} has nan gradient. Setting it to zero.")
                p.grad[torch.isnan(p.grad)] = 0.


def train_mask(
    model: HookedTransformer,
    mask: BasicMask,
    retain_dataloader: DataLoader,
    forget_dataloader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    max_gpu_batch_size: int = 32,
    alpha: float = 0.2,
    beta: float = 1.0,
    clip_grad: float = 1.0,
):
    # Initialize optimizer
    mask = mask.cuda()
    optimizer = torch.optim.AdamW(mask.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
    # Cycle dataloaders
    retain_dataloader = cycle(retain_dataloader)
    forget_dataloader = cycle(forget_dataloader)
    # Train a sparse mask
    for epoch in tqdm(range(n_epochs)):
        # Sample batches
        retain_batch = next(retain_dataloader)
        forget_batch = next(forget_dataloader)
        # Reset grad
        optimizer.zero_grad()
        # Compute normal loss over retain
        retain_batch_size = retain_batch['tokens'].shape[0]
        retain_loss = 0
        for start_idx in range(0, retain_batch_size, max_gpu_batch_size):
            # Subsample batch
            sub_batch = get_minibatch(retain_batch, start_idx, max_gpu_batch_size)
            sub_batch_length = sub_batch["tokens"].shape[0]
            tokens = sub_batch["tokens"].cuda()
            tokens_mask = sub_batch["mask"][:, 1:].cuda()
            # Compute loss
            #with torch.autocast(device_type="cuda"):
            logits = model(tokens[:, :-1])
            labels = tokens[:, 1:]
            loss = (sub_batch_length/retain_batch_size) * F.cross_entropy(logits[tokens_mask], labels[tokens_mask])
            loss.backward()
            retain_loss += loss.item()
        # Compute flipped loss over forget
        forget_batch_size = forget_batch['tokens'].shape[0]
        forget_loss = 0
        for start_idx in range(0, forget_batch_size, max_gpu_batch_size):
            # Subsample batch
            sub_batch = get_minibatch(forget_batch, start_idx, max_gpu_batch_size)
            sub_batch_length = sub_batch["tokens"].shape[0]
            tokens = sub_batch["tokens"].cuda()
            tokens_mask = sub_batch["mask"][:, 1:].cuda()
            # Compute loss
            logits = model(tokens[:, :-1])
            labels = tokens[:, 1:]
            loss = alpha * (sub_batch_length/forget_batch_size) *  log_1_minus_p_loss(logits[tokens_mask], labels[tokens_mask])
            loss.backward()
            forget_loss += loss.item()
        # Add sparsity loss and backprop
        loss = beta * mask.regularization_loss()
        loss.backward()
        reg_loss = loss.item()
        # Step and log
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(mask.parameters(), clip_grad)
        # zero_nan_grads(mask)
        optimizer.step()
        mask.on_step_end()
        print(f"Retain Loss: {retain_loss:.3f}, Forget Loss: {forget_loss:.3f}, Reg Loss: {reg_loss:.3f}")
        scheduler.step()
        
        