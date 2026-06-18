"""
experts.py — Dual-Branch Expert Architecture (Section 3.2).
Implements the MoE modules for both shared and domain self-adaptive branches.
Each MoE module: a router + M low-rank experts (Eq. 1-2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LowRankExpert(nn.Module):
    """Single low-rank expert: down-project → activation → up-project (Eq. 1)."""

    def __init__(self, dim, rank, act=nn.GELU):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.act = act()
        # zero-init up projection so expert output starts at zero
        nn.init.zeros_(self.up.weight)

    def forward(self, z):
        return self.up(self.act(self.down(z)))


class MoEModule(nn.Module):
    """
    Mixture-of-Experts module with M experts and a learned router (Eq. 2).
    """

    def __init__(self, dim, rank, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            LowRankExpert(dim, rank) for _ in range(num_experts)
        ])
        # router: linear layer mapping input → gating scores
        self.router = nn.Linear(dim, num_experts, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, z):
        """
        z: [B, N, D]  (batch, tokens, dim)
        returns: [B, N, D]
        """
        # compute gating scores → softmax weights (Eq. 2)
        scores = self.router(z)                         # [B, N, M]
        alpha = F.softmax(scores, dim=-1)               # [B, N, M]

        # weighted sum of expert outputs
        out = torch.zeros_like(z)
        for i, expert in enumerate(self.experts):
            out = out + alpha[..., i:i+1] * expert(z)   # [B, N, D]
        return out


class DualBranchExpert(nn.Module):
    """
    Dual-Branch Expert block inserted into each ViT FFN (Fig. 2).
    Contains:
      - A domain-shared MoE (always active, always trainable)
      - A pool of domain self-adaptive MoE modules (one per detected domain;
        only the active one is trainable, rest are frozen)
    Output: Z_out = Z + λ * Y_shared + (1-λ) * Y_domain  (Eq. 5)
    """

    def __init__(self, dim, shared_rank=32, domain_rank=16,
             num_experts=2, fusion_lambda=0.5, use_shared=True):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.fusion_lambda = fusion_lambda
        self.use_shared = use_shared

        self.domain_rank = domain_rank

        # shared expert branch (Eq. 3)
        self.shared_moe = MoEModule(dim, shared_rank, num_experts)

        # domain self-adaptive expert pool (Eq. 4)
        # starts empty; modules added dynamically by expand_domain()
        self.domain_moes = nn.ModuleList()

        # currently active domain index (-1 = none)
        self.active_domain = -1

    @property
    def num_domains(self):
        return len(self.domain_moes)

    def expand_domain(self):
        """
        Add a new domain self-adaptive MoE module.
        Returns the new domain index.
        """

        new_moe = MoEModule(dim=self.dim, rank=self.domain_rank, 
                     num_experts=self.num_experts)
        # move to same device as shared_moe
        device = next(self.shared_moe.parameters()).device
        new_moe = new_moe.to(device)
        self.domain_moes.append(new_moe)
        return len(self.domain_moes) - 1

    def set_active_domain(self, domain_id):
        """
        Activate domain_id's MoE and freeze all others.
        """
        self.active_domain = domain_id
        for i, moe in enumerate(self.domain_moes):
            requires_grad = (i == domain_id)
            for p in moe.parameters():
                p.requires_grad = requires_grad

    def forward(self, z):
        if self.use_shared:
            lam = self.fusion_lambda
            y_shared = self.shared_moe(z)
        else:
            lam = 0.0
            y_shared = torch.zeros_like(z)
    
        if self.active_domain >= 0 and self.active_domain < len(self.domain_moes):
            y_domain = self.domain_moes[self.active_domain](z)
        else:
            y_domain = torch.zeros_like(z)
    
        z_out = z + lam * y_shared + (1.0 - lam) * y_domain
        return z_out


class DualBranchExpertManager:
    """
    Manages all DualBranchExpert blocks across ViT layers.
    Provides unified interface for:
      - expanding to a new domain (adds MoE in every block)
      - setting active domain (freezes/unfreezes across all blocks)
      - collecting trainable parameters
    """

    def __init__(self):
        self.blocks = []     # list of DualBranchExpert modules

    def register(self, block: DualBranchExpert):
        self.blocks.append(block)

    def expand_domain(self):
        """Add a new domain expert across all blocks. Returns domain index."""
        idx = None
        for block in self.blocks:
            idx = block.expand_domain()
        return idx

    def set_active_domain(self, domain_id):
        """Activate a domain across all blocks."""
        for block in self.blocks:
            block.set_active_domain(domain_id)

    @property
    def num_domains(self):
        return self.blocks[0].num_domains if self.blocks else 0

    def trainable_parameters(self):
        params = []
        for block in self.blocks:
            if block.use_shared:
                for p in block.shared_moe.parameters():
                    if p.requires_grad:
                        params.append(p)
            if 0 <= block.active_domain < len(block.domain_moes):
                for p in block.domain_moes[block.active_domain].parameters():
                    if p.requires_grad:
                        params.append(p)
        return params

    def all_expert_parameters(self):
        """All expert parameters (for state_dict save/load)."""
        params = {}
        for i, block in enumerate(self.blocks):
            params[f"block_{i}_shared"] = block.shared_moe.state_dict()
            for j, moe in enumerate(block.domain_moes):
                params[f"block_{i}_domain_{j}"] = moe.state_dict()
        return params
