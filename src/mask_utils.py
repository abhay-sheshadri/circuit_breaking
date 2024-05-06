from functools import partial

import torch
from torch import nn
from transformer_lens import HookedTransformer


class MaskRoot(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def _get_mask_hook(self, mask):
        raise NotImplementedError()

    def regularization_loss(self):
        raise NotImplementedError() 


class BasicMask(MaskRoot):
    
    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None,

        disable_model_grads=True
    ):
        super().__init__()
        model.reset_hooks()
        if disable_model_grads:
            # set all model params to not require grads
            for param in model.parameters():
                param.requires_grad_(False)
        
        self.masks = nn.ParameterDict()
        self.mask_masks = {} # A mask for our masks
        self.binarized = False
        
        self.start_layer = start_layer
        if end_layer is None:
            self.end_layer = model.cfg.n_layers
        else:
            self.end_layer = end_layer

    def regularization_loss(self):
        # Compute the L1 of the mask
        total_loss = 0
        for layer in self.masks:
            total_loss += (1 - self.masks[layer]).mean()
        return total_loss
    
    def undiscretize(self,):
        # Set everything in each maskmask to false
        for key in self.mask_masks:
            self.mask_masks[key].fill_(False)
        # Set binarized to false
        self.binarized = False

    def discretize_threshold(self, threshold):
        # Undiscretize
        self.undiscretize()
        # Mask out everything less sthan threshold
        for key, tensor in self.masks.items():
            self.mask_masks[key] = tensor <= threshold
        # Set binarized to true
        self.binarized = True

    def discretize_topk(self, k):
        # Undiscretize
        self.undiscretize()
        # If k == 0, everything is false
        if k != 0:
            # Flatten all tensors and concatenate them into one big tensor to find the top 1% value
            all_values = torch.cat([tensor.data.flatten() for tensor in self.masks.values()])
            threshold = all_values.kthvalue(k).values
            # Mask out everything less than threshold
            for key, tensor in self.masks.items():
                self.mask_masks[key] = tensor <= threshold
        # Set binarized to true
        self.binarized = True
    
    def discretize_topk_percent(self, percentile):
        # Undiscretize
        self.undiscretize()
        # If k == 0, everything is false
        if k != 0:
            # Flatten all tensors and concatenate them into one big tensor to find the top 1% value
            all_values = torch.cat([tensor.data.flatten() for tensor in self.masks.values()])
            k = int(percentile  * all_values.numel())
            threshold = all_values.kthvalue(k).values
            # Mask out everything less than threshold
            for key, tensor in self.masks.items():
                self.mask_masks[key] = tensor <= threshold
        # Set binarized to true
        self.binarized = True

    def num_masked(self):
        if not self.binarized:
            return None
        else:
            total = 0
            for mask in self.mask_masks.values():
                total += torch.sum(mask).item()
            return total
    
    def on_step_end(self):
        # Clip the masks at every step end
        with torch.no_grad():
            for layer in self.masks:
                self.masks[layer].clamp_(0, 1)

