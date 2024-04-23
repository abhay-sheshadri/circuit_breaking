from functools import partial

import torch
from torch import nn
from transformer_lens import HookedTransformer

from .mask_utils import *


class MLPHiddenMask(BasicMask):
    
    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__(
            model=model,
            start_layer=start_layer,
            end_layer=end_layer
        )

        for layer in range(self.start_layer, self.end_layer):
            # Initialize mask
            mask = nn.Parameter(torch.ones(model.cfg.d_mlp))
            mask.requires_grad_(True)
            self.masks[f"mlp_{layer}"] = mask
            self.mask_masks[f"mlp_{layer}"] = torch.zeros_like(mask).to(torch.bool)
            
            # Add the mask as a hook over the component
            mlp_hook = model.blocks[layer].mlp.hook_post
            temp_hook = self._get_mask_hook(layer)
            mlp_hook.add_hook(temp_hook)
        
    def _get_mask_hook(self, layer_idx):
        # Return the hook function
        def hook_fn(acts, hook):
            mlp_mask = self.masks[f"mlp_{layer_idx}"].to(acts.dtype)
            if self.binarized:
                mlp_mask_mask = self.mask_masks[f"mlp_{layer_idx}"]
                mlp_mask[mlp_mask_mask] = 0
                mlp_mask[~mlp_mask_mask] = 1
            return acts * mlp_mask
        return hook_fn


class MLPOutputMask(BasicMask):

    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__(
            model=model,
            start_layer=start_layer,
            end_layer=end_layer
        )

        for layer in range(self.start_layer, self.end_layer):
            # Initialize mask
            mask = nn.Parameter(torch.ones(model.cfg.d_model))
            mask.requires_grad_(True)
            self.masks[f"mlp_{layer}"] = mask
            self.mask_masks[f"mlp_{layer}"] = torch.zeros_like(mask).to(torch.bool)
            
            # Add the mask as a hook over the component
            mlp_hook = model.blocks[layer].hook_mlp_out
            temp_hook = self._get_mask_hook(layer)
            mlp_hook.add_hook(temp_hook)
        
    def _get_mask_hook(self, layer_idx):
        # Return the hook function
        def hook_fn(acts, hook):
            mlp_mask = self.masks[f"mlp_{layer_idx}"].to(acts.dtype)
            if self.binarized:
                mlp_mask_mask = self.mask_masks[f"mlp_{layer_idx}"]
                mlp_mask[mlp_mask_mask] = 0
                mlp_mask[~mlp_mask_mask] = 1
            return acts * mlp_mask
        return hook_fn


class MLPOutputSVDMask(BasicMask):

    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__(
            model=model,
            start_layer=start_layer,
            end_layer=end_layer
        )
        
        self.svd_projections = {}

        for layer in range(self.start_layer, self.end_layer):
            # Initialize mask
            mask = nn.Parameter(torch.ones(model.cfg.d_model))
            mask.requires_grad_(True)
            self.masks[f"mlp_{layer}"] = mask
            self.mask_masks[f"mlp_{layer}"] = torch.zeros_like(mask).to(torch.bool)
            
            # Add the mask as a hook over the component
            mlp_hook = model.blocks[layer].hook_mlp_out

            mlp_out_weight = model.blocks[layer].mlp.W_out
            mlp_out_weight_f32 = mlp_out_weight.float()
            _, _, V_T = torch.svd(mlp_out_weight_f32)

            # V_T, U take SVD components and output features
            self.svd_projections[f"mlp_{layer}"] = V_T.to(mlp_out_weight.dtype)

            temp_hook = self._get_mask_hook(layer)
            mlp_hook.add_hook(temp_hook)
    
    def _get_mask_hook(self, layer_idx):
        # Return the hook function
        def hook_fn(acts, hook):
            mlp_mask = self.masks[f"mlp_{layer_idx}"].to(acts.dtype)
            if self.binarized:
                mlp_mask_mask = self.mask_masks[f"mlp_{layer_idx}"]
                mlp_mask[mlp_mask_mask] = 0
                mlp_mask[~mlp_mask_mask] = 1
            
            projected_acts = torch.einsum("i j, ... i -> ... j", self.svd_projections[f"mlp_{layer_idx}"], acts)
            projected_acts *= mlp_mask
            unprojected_acts = torch.einsum("i j, ... j -> ... i", self.svd_projections[f"mlp_{layer_idx}"], projected_acts)

            return unprojected_acts
        return hook_fn


class NeuronLevelMask(BasicMask):
    # MLPInput over MLPs, SVD over W_o
    pass

class FeatureLevelMask(MaskRoot):
    # Use an SAE as a mask
    pass