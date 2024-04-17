from functools import partial

import torch
from torch import nn
from transformer_lens import HookedTransformer


class BasicMask(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def _get_mask_hook(self, mask):
        raise NotImplementedError()

    def regularization_loss(self):
        raise NotImplementedError() 


class NeuronLevelMask(BasicMask):
    
    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__()
        model.reset_hooks()
        
        self.masks = nn.ParameterDict()
        
        if end_layer is None:
            end_layer = model.cfg.n_layers

        for layer in range(start_layer, end_layer):
            # Initialize mask
            mask = nn.Parameter(torch.ones(model.cfg.d_mlp))
            mask.requires_grad_(True)
            self.masks[f"mlp_{layer}"] = mask

            # Add the mask as a hook over the component
            mlp_hook = model.blocks[layer].mlp.hook_post
            temp_hook = self._get_mask_hook(mask)
            mlp_hook.add_hook(temp_hook)
        
    def _get_mask_hook(self, mask):
        def hook_fn(acts, hook):
            return acts * mask#[None, None, :]
        return hook_fn
    
    def on_step_end(self):
        with torch.no_grad():
            for layer in self.masks:
                self.masks[layer].clamp_(0, 1)
            
    def regularization_loss(self):
        total_loss = 0
        for layer in self.masks:
            total_loss += torch.sqrt(1 - self.masks[layer] + 1e-1).mean()
        return total_loss
    

class FeatureLevelMask(BasicMask):
    pass