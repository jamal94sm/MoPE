"""
tent.py — TENT: Fully Test-Time Adaptation by Entropy Minimization.

Wang et al., ICLR 2021.
Aligned with: https://github.com/DequanWang/tent

Core idea: At test time, update only the affine parameters (weight, bias)
of BatchNorm layers by minimizing the entropy of model predictions.
BN running statistics are disabled — only batch statistics are used.

Usage:
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    tented_model = tent.Tent(model, optimizer)
    outputs = tented_model(inputs)  # infers and adapts
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Save initial state for episodic reset
        # Note: if never reset (continual adaptation), this copy is unused
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: original TENT only collects BatchNorm2d. We also include
    BatchNorm1d for models like iResNet100 that use BN1d in the
    final embedding layer.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent.

    Following the original TENT implementation:
    1. Set entire model to train mode (needed for BN to use batch stats)
    2. Disable gradients for all parameters
    3. Re-enable gradients for BN affine parameters only
    4. Disable running stats tracking — force batch statistics

    Note: original only handles BatchNorm2d. We also handle BatchNorm1d
    for iResNet100 compatibility.
    """
    # Train mode — required so BN uses batch statistics
    model.train()

    # Disable all gradients
    model.requires_grad_(False)

    # Enable BN affine params + force batch statistics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            # Force use of batch stats in both train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def check_model(model):
    """Check model for compatibility with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"

    param_grads = [p.requires_grad for p in model.parameters()]
    has_any = any(param_grads)
    has_all = all(param_grads)
    assert has_any, "tent needs params to update: check which require grad"
    assert not has_all, "tent should not update all params: check which require grad"

    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
                 for m in model.modules())
    assert has_bn, "tent needs normalization for its optimization"


# ══════════════════════════════════════════════════════════════
#  BNA: Batch Norm Adaptation (no gradients, no head)
# ══════════════════════════════════════════════════════════════

def configure_model_bna(model):
    """
    Configure model for BNA (Batch Norm Adaptation).

    1. Set model to eval mode
    2. Reset BN running stats
    3. Set BN to train mode (use batch stats + update running stats)
    4. Freeze all parameters (no gradients at all)
    """
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.train()  # use batch stats + accumulate into running stats

    return model


@torch.no_grad()
def forward_bna(x, model):
    """Forward pass for BNA. No gradients, just updates BN running stats."""
    return model(x)


# ══════════════════════════════════════════════════════════════
#  Contrastive TENT: Entropy + Contrastive on augmented pairs
# ══════════════════════════════════════════════════════════════

import torch.nn.functional as F
from torchvision import transforms


def get_tta_augmentation(img_size=112):
    """
    Mild augmentation for contrastive TTA.
    Enough to create a different view, not so much it destroys identity.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0),
                                      ratio=(0.95, 1.05)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                    saturation=0.05, hue=0.02),
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ], p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def augment_batch(x, aug_transform, mean=0.5, std=0.5):
    """
    Apply augmentation to a normalized batch.
    Denormalize → augment (PIL-like ops on tensors) → renormalize.

    x: (B, C, H, W) normalized tensor
    Returns: augmented tensor, same shape
    """
    # Denormalize to [0, 1]
    x_denorm = x * std + mean
    x_denorm = x_denorm.clamp(0, 1)

    # Augment each image
    augmented = []
    for i in range(x_denorm.shape[0]):
        img = x_denorm[i]  # (C, H, W) tensor in [0, 1]
        img_aug = aug_transform(img)  # torchvision transforms work on tensors
        augmented.append(img_aug)

    x_aug = torch.stack(augmented).to(x.device)

    # Renormalize
    x_aug = (x_aug - mean) / std
    return x_aug


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    z1, z2: (B, D) L2-normalized embeddings (anchor, positive)
    Each z1[i] is positive with z2[i], negative with all others.

    Returns: scalar loss
    """
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Similarity matrix: (2B, 2B)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.T) / temperature   # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),  # z1[i] → z2[i]
        torch.arange(0, B, device=z.device),        # z2[i] → z1[i]
    ])

    return F.cross_entropy(sim, labels)


class ContrastiveTent(nn.Module):
    """
    TENT + Contrastive loss on augmented views.

    Loss = entropy(logits) + λ * NT-Xent(z_orig, z_aug)

    Entropy pushes BN toward confident predictions.
    Contrastive pulls embeddings of augmented views together,
    improving feature consistency across domain variations.
    """
    def __init__(self, model, optimizer, aug_transform,
                 contrastive_lambda=1.0, contrastive_temp=0.5,
                 steps=1, episodic=False, use_entropy=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.steps = steps
        self.episodic = episodic
        self.use_entropy = use_entropy
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        # Original forward
        logits = self.model(x)

        # Get embeddings for contrastive
        if hasattr(self.model, 'get_embeddings'):
            z_orig = self.model.get_embeddings(x)
        elif hasattr(self.model, 'backbone'):
            z_orig = self.model.backbone(x)
        else:
            z_orig = self.model(x)

        # Augmented forward
        x_aug = augment_batch(x, self.aug_transform)
        if hasattr(self.model, 'get_embeddings'):
            z_aug = self.model.get_embeddings(x_aug)
        elif hasattr(self.model, 'backbone'):
            z_aug = self.model.backbone(x_aug)
        else:
            z_aug = self.model(x_aug)

        # Losses
        info = {}

        total_loss = torch.tensor(0.0, device=x.device)

        if self.use_entropy:
            ent_loss = softmax_entropy(logits).mean(0)
            total_loss = total_loss + ent_loss
            info["entropy"] = ent_loss.item()

        con_loss = nt_xent_loss(z_orig, z_aug, self.contrastive_temp)
        total_loss = total_loss + self.contrastive_lambda * con_loss
        info["contrastive"] = con_loss.item()
        info["total"] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
