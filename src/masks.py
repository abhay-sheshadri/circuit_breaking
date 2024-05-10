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

def convert_param_name(component, inverse=False):
    """To solve KeyError: 'parameter name can\'t contain "."' Replace . with &"""
    if inverse:
        return component.replace("&", ".")
    else:
        return component.replace(".", "&")


class NeuronLevelMask(BasicMask):

    def create_frozen_unfrozen_params(self, heads_to_mask, component_sample_cache):
        assert component_sample_cache.shape[-2] == self.model_cfg.n_heads

        # Create frozen and unfrozen parameters
        unfrozen_heads = torch.nn.Parameter(torch.zeros((component_sample_cache.shape[-2], 1), dtype=torch.bool), requires_grad=False)
        unfrozen_heads[heads_to_mask] = True
        frozen_heads = torch.nn.Parameter(~unfrozen_heads, requires_grad=False)
        
        trainable_mask = torch.nn.Parameter(torch.ones((component_sample_cache.shape[-2], component_sample_cache.shape[-1])), requires_grad=True)

        return frozen_heads.float(), unfrozen_heads.float(), trainable_mask

    def convert_param_name(self, component, inverse=False):
        """To solve KeyError: 'parameter name can\'t contain "."' Replace . with &"""
        if inverse:
            return component.replace("&", ".")
        else:
            return component.replace(".", "&")

    def __init__(
        self,
        model: HookedTransformer,
        components,
        component_heads=None,
        sample_input_for_shapes="test"
    ):
        """
        Outpast over arbitrary components. Can apply to attention head q, k, v, result, mlp post, mlp pre.

        components should be list or set of strings which are keys of model.hook_dict, being one of {"blocks.l.mlp.hook_pre", "blocks.l.mlp.hook_post", "blocks.l.attn.hook_q", "blocks.l.attn.hook_k", "blocks.l.attn.hook_v", "blocks.l.attn.hook_result"}
        component_heads should be a dictionary of keys=components to values=list of heads to mask. All keys should be components with attn in them.

        Hacky: to determine the shape of each mask, uses a sample input to the model.
        """
        super().__init__(
            model=model,
        )

        self.model_cfg = model.cfg
        self.components = components
        self.component_heads = component_heads
        # self.attn_masks = nn.ParameterDict()
        self.attn_mask_frozen = nn.ParameterDict()
        self.attn_mask_unfrozen = nn.ParameterDict()
        with torch.no_grad():
            _, sample_cache = model.run_with_cache(sample_input_for_shapes)

        for component in components:
            assert component in model.hook_dict and component in sample_cache, f"Component {component} not found in model.hook_dict or sample_cache"
            # Initialize mask
            if "attn" in component:
                assert component_heads is not None and component in component_heads and component_heads[component] is not None
                # assert "attn" in component, f"Component {component} has heads specified but is not an attention component"

                frozen_heads, unfrozen_heads, trainable_mask = self.create_frozen_unfrozen_params(component_heads[component], sample_cache[component])
                trainable_mask.requires_grad_(True)
                # frozen_heads.requires_grad_(False)
                # unfrozen_heads.requires_grad_(False)
                self.masks[self.convert_param_name(component)] = trainable_mask
                self.mask_masks[self.convert_param_name(component)] = torch.zeros_like(trainable_mask).to(torch.bool)
                self.attn_mask_frozen[self.convert_param_name(component)] = frozen_heads
                self.attn_mask_unfrozen[self.convert_param_name(component)] = unfrozen_heads
                
                temp_hook = self._get_attn_mask_hook(component)
            # if "attn" not in component: # mlps
            else:
                mask = nn.Parameter(torch.ones(sample_cache[component].shape[-1]))
                mask.requires_grad_(True)
                self.masks[self.convert_param_name(component)] = mask
                self.mask_masks[self.convert_param_name(component)] = torch.zeros_like(mask).to(torch.bool)
            
                # Add the mask as a hook over the component
                temp_hook = self._get_mask_hook(component)

            model.hook_dict[component].add_hook(temp_hook)

        self.attn_mask_frozen.requires_grad_(False)
        self.attn_mask_unfrozen.requires_grad_(False)


    def _get_mask_hook(self, component):
        # Return the hook function
        def hook_fn(acts, hook):
            component_mask = self.masks[self.convert_param_name(component)].to(acts.dtype)
            if self.binarized:
                component_mask_mask = self.mask_masks[self.convert_param_name(component)]
                component_mask[component_mask_mask] = 0
                component_mask[~component_mask_mask] = 1
            return acts * component_mask
        return hook_fn

    def _get_attn_mask_hook(self, component):
        def hook_fn(acts, hook):
            frozen_heads = self.attn_mask_frozen[self.convert_param_name(component)]
            unfrozen_heads = self.attn_mask_unfrozen[self.convert_param_name(component)]
            trainable_mask = self.masks[self.convert_param_name(component)]
            
            # print(f"{frozen_heads.device=}, {unfrozen_heads.device=}, {trainable_mask.device=}")
            trainable_mask = trainable_mask.to(acts.dtype)
            frozen_heads = frozen_heads.to(acts.dtype)
            unfrozen_heads = unfrozen_heads.to(acts.dtype)
            
            if self.binarized:
                trainable_mask_mask = self.mask_masks[self.convert_param_name(component)]
                trainable_mask[trainable_mask_mask] = 0
                trainable_mask[~trainable_mask_mask] = 1
            
            # print(f"{frozen_heads.device}, {unfrozen_heads.device}, {trainable_mask.device}, {acts.device}, {self.mask_masks[self.convert_param_name(component)].device}")
            component_mask = frozen_heads + trainable_mask * unfrozen_heads
            # print(f"Component mask: {component_mask.device=}, {acts.device=}")
            return acts * component_mask

        return hook_fn

    def regularization_loss(self):
        # Compute the L1 of the mask
        total_loss = 0
        # mlps first
        for component in self.masks:
            if "attn" in component:
                assert component in self.attn_mask_unfrozen and component in self.attn_mask_frozen
                total_loss += (1 - self.masks[component] * self.attn_mask_unfrozen[component]).mean()
            else:
                total_loss += (1 - self.masks[component]).mean()
        return total_loss
    
    def discretize_topk_percent(self, percentile):
        # Undiscretize
        self.undiscretize()
        # If k == 0, everything is false
        if k != 0:
            # Flatten all tensors and concatenate them into one big tensor to find the top 1% value
            # all_values = torch.cat([tensor.data.flatten() for tensor in self.masks.values()])
            all_values = []
            for component in self.masks:
                if "attn" in component:
                    assert component in self.attn_mask_unfrozen and component in self.attn_mask_frozen

                    # add components of the mask val that are trainable
                    trainable_head_indices = self.attn_mask_unfrozen[component].data.flatten().nonzero().flatten()
                    print(f"For {component=}, {trainable_head_indices=}")
                    all_values.append(self.masks[component][trainable_head_indices].data.flatten())
                else:
                    all_values.append(self.masks[component].data.flatten())
                # all_values.append(self.masks[component].data.flatten())

            k = int(percentile  * all_values.numel())
            threshold = all_values.kthvalue(k).values
            # Mask out everything less than threshold
            for key, tensor in self.masks.items():
                self.mask_masks[key] = tensor <= threshold
        # Set binarized to true
        self.binarized = True


class FeatureLevelMask(MaskRoot):
    # Use an SAE as a mask
    pass