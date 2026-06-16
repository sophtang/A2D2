import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# additional sigmoid head
# ------------------------------------------------------------
class RemaskingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.proj1(h)
        h = self.act(h)
        h = self.proj2(h)
        return h


class InsertionQualityHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.proj1(h)
        h = self.act(h)
        h = self.proj2(h)
        return h


class RemaskingAnyOrder(nn.Module):
    """Remasking adapter for AnyOrderMaskInsertionFlow models."""
    def __init__(self, backbone: nn.Module, d_model: int, insertion_planner: bool = False):
        super().__init__()
        
        # Store backbone as non-module attribute to avoid circular reference in module tree
        # Use object.__setattr__ to bypass nn.Module's __setattr__ which registers modules
        object.__setattr__(self, 'backbone', backbone)
        self.d_model = d_model
        self.insertion_planner = insertion_planner
        self.remasking_head = RemaskingHead(d_model)
        
        if insertion_planner:
            self.insertion_head = InsertionQualityHead(d_model)
        
    def forward(self, indices: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Forward pass for remasking training.

        Args:
            indices: Token indices [batch_size, seq_len]
            t: Timesteps [batch_size]
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dict with 'logits', 'remasking_conf', and optionally 'insertion_conf' keys
        """
        # Single backbone pass returning both prediction and post-block features.
        # features has shape [B, L+1, hidden]; the remasking/insertion heads use
        # the same [B, L, hidden] slice that get_hidden_states would have returned.
        prediction, features = self.backbone(indices, t, return_features=True)
        hidden_states = features[:, :-1]

        remasking_conf = self.remasking_head(hidden_states)
        token_logits = prediction.token_logits

        result = {"logits": token_logits, "remasking_conf": remasking_conf}

        if self.insertion_planner:
            insertion_conf = self.insertion_head(hidden_states)
            result["insertion_conf"] = insertion_conf

        return result
    
    def get_hidden_states(self, indices: torch.Tensor, t: torch.Tensor):
        """
        Get hidden states and logits for adapter training.
        
        Args:
            indices: Token indices [batch_size, seq_len]
            t: Timesteps [batch_size]
            
        Returns:
            Tuple of (token_logits, hidden_states, conditioning)
        """
        return self.backbone.get_hidden_states(indices, t)
    
    @property
    def device(self):
        return next(self.backbone.parameters()).device
