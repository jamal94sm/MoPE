"""
model.py — ViT backbone with Dual-Branch Expert insertion.
Loads a pretrained ViT-Base from timm, freezes it, and inserts
DualBranchExpert modules into each transformer block's FFN.
"""

import torch
import torch.nn as nn
import timm
from experts import DualBranchExpert, DualBranchExpertManager


class ExpertViT(nn.Module):
    """
    ViT-Base with frozen backbone + dual-branch expert modules
    inserted as bypass in each transformer block's FFN.

    Architecture per block:
        z -> LN -> MHSA -> + (residual) -> LN -> [MLP || DualBranchExpert] -> + (residual)

    The DualBranchExpert is added as a bypass alongside the frozen MLP:
        block_out = z + MLP(LN(z)) + DualBranchExpert(LN(z))
    which is equivalent to Eq. 5 since DualBranchExpert internally adds
    the residual z.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # load pretrained backbone
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True)

        # freeze entire backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # get hidden dimension from the backbone
        if hasattr(self.backbone, 'embed_dim'):
            dim = self.backbone.embed_dim                # ViT
        else:
            dim = self.backbone.num_features

        # insert dual-branch experts into each transformer block
        self.expert_manager = DualBranchExpertManager()

        blocks = self._get_blocks()
        for block in blocks:
            effective_domain_rank = cfg.domain_high_rank if cfg.vida_domain else cfg.domain_rank
            expert = DualBranchExpert(
                dim=dim,
                shared_rank=cfg.shared_rank,
                domain_rank=cfg.domain_rank,
                num_experts=cfg.num_experts_per_moe,
                fusion_lambda=cfg.fusion_lambda,
                use_shared=cfg.use_shared_expert,
            )
            # register the hook or store the expert
            self.expert_manager.register(expert)

        # store experts as a ModuleList so they get proper device handling
        self.expert_modules = nn.ModuleList(self.expert_manager.blocks)

        # store original forward functions for hooks
        self._install_hooks(blocks)

    def _get_blocks(self):
        """Get the list of transformer blocks from the timm ViT."""
        if hasattr(self.backbone, 'blocks'):
            return list(self.backbone.blocks)
        elif hasattr(self.backbone, 'layers'):
            return list(self.backbone.layers)
        raise ValueError(f"Cannot find blocks in {type(self.backbone)}")

    def _install_hooks(self, blocks):
        """
        Install forward hooks on each block to add the expert bypass.
        After the block's original forward, we add the expert output
        computed from the pre-FFN features.
        """
        self._block_inputs = {}

        for i, (block, expert) in enumerate(zip(blocks, self.expert_modules)):
            # save original mlp forward
            original_mlp_forward = block.mlp.forward

            # create a wrapper that also runs the expert bypass
            def make_hook(expert_mod, orig_fwd):
                def hooked_mlp_forward(x):
                    # original MLP output (frozen)
                    mlp_out = orig_fwd(x)
                    # expert bypass: DualBranchExpert expects [B, N, D]
                    # but internally does z_out = z + λ*shared + (1-λ)*domain
                    # We want bypass, so subtract z from expert output:
                    expert_out = expert_mod(x)   # = x + λ*shared + (1-λ)*domain
                    bypass = expert_out - x      # = λ*shared + (1-λ)*domain
                    return mlp_out + bypass
                return hooked_mlp_forward

            block.mlp.forward = make_hook(expert, original_mlp_forward)

    def forward(self, x):
        """Standard forward through the backbone (with expert hooks active)."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x):
        """Forward pass returning probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def expand_domain(self):
        """Add a new domain expert across all blocks."""
        return self.expert_manager.expand_domain()

    def set_active_domain(self, domain_id):
        """Set active domain across all blocks."""
        self.expert_manager.set_active_domain(domain_id)

    def get_trainable_params(self):
        """Get currently trainable parameters for optimizer."""
        return self.expert_manager.trainable_parameters()

    @property
    def num_domains(self):
        return self.expert_manager.num_domains


def build_model(cfg):
    """Build and return the ExpertViT model."""
    model = ExpertViT(cfg)
    model = model.to(cfg.device)
    model.eval()            # backbone stays in eval mode
    return model
