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
        self.mask_masks = {} # A mask for our masks
        self.binarized = False
        
        if end_layer is None:
            end_layer = model.cfg.n_layers

        for layer in range(start_layer, end_layer):
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
    
    def on_step_end(self):
        # Clip the masks at every step end
        with torch.no_grad():
            for layer in self.masks:
                self.masks[layer].clamp_(0, 1)
            
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

class MLPOutputMask(BasicMask):

    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__()
        model.reset_hooks()
        
        self.masks = nn.ParameterDict()
        self.mask_masks = {} # A mask for our masks
        self.binarized = False
        
        if end_layer is None:
            end_layer = model.cfg.n_layers

        for layer in range(start_layer, end_layer):
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
    
    def on_step_end(self):
        # Clip the masks at every step end
        with torch.no_grad():
            for layer in self.masks:
                self.masks[layer].clamp_(0, 1)
            
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

class SVDMask(BasicMask):

    def __init__(
        self,
        model: HookedTransformer,
        start_layer: int = 0,
        end_layer: int = None
    ):
        super().__init__()
        model.reset_hooks()
        
        self.svd_projections = {}
        self.masks = nn.ParameterDict()
        self.mask_masks = {} # A mask for our masks
        self.binarized = False
        
        if end_layer is None:
            end_layer = model.cfg.n_layers

        for layer in range(start_layer, end_layer):
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
            print(V_T.shape)

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

    def on_step_end(self):
        # Clip the masks at every step end
        with torch.no_grad():
            for layer in self.masks:
                self.masks[layer].clamp_(0, 1)
            
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

class FeatureLevelMask(BasicMask):
    pass