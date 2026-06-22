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


def configure_model_safe(model):
    """
    Configure model for gradient-based TTA on small datasets.

    Unlike TENT's configure_model which nulls running stats (forcing
    pure batch stats), this preserves running stats so they accumulate
    from target data via EMA. At eval time, the adapted running stats
    are used for consistent normalization.

    1. Set entire model to train mode (BN uses batch stats + updates EMA)
    2. Freeze all parameters
    3. Re-enable gradients for BN affine params (γ, β)
    4. KEEP running stats (don't null them)
    """
    model.train()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            # Keep track_running_stats = True (default)
            # Keep existing running_mean / running_var
            # BN in train mode: uses batch stats for forward pass
            #   AND updates running_mean/var via EMA
            # At eval time: uses adapted running stats for consistency

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


def fft_amplitude_swap(x, beta=0.5):
    """
    FFT-based augmentation: swap low-frequency amplitude of each sample
    with a random other sample in the batch.

    Keeps phase (structure/identity) intact, changes amplitude
    (style/appearance). Since all samples are from the same domain,
    this creates valid within-domain variations.

    x: (B, C, H, W) tensor in normalized space
    beta: fraction of low-frequency band to swap (0=none, 1=all)
    Returns: augmented tensor, same shape
    """
    B, C, H, W = x.shape
    # Random permutation for pairing
    perm = torch.randperm(B, device=x.device)
    x_partner = x[perm]

    # FFT
    fft_x = torch.fft.fft2(x, dim=(-2, -1))
    fft_p = torch.fft.fft2(x_partner, dim=(-2, -1))

    amp_x = torch.abs(fft_x)
    phase_x = torch.angle(fft_x)
    amp_p = torch.abs(fft_p)

    # Low-frequency mask (center of spectrum)
    cy, cx = H // 2, W // 2
    rh = int(H * beta * 0.5)
    rw = int(W * beta * 0.5)

    # Shift so DC is at center
    amp_x_shifted = torch.fft.fftshift(amp_x, dim=(-2, -1))
    amp_p_shifted = torch.fft.fftshift(amp_p, dim=(-2, -1))

    # Swap low-frequency amplitudes
    amp_mixed = amp_x_shifted.clone()
    y1, y2 = max(0, cy - rh), min(H, cy + rh)
    x1, x2 = max(0, cx - rw), min(W, cx + rw)
    amp_mixed[:, :, y1:y2, x1:x2] = amp_p_shifted[:, :, y1:y2, x1:x2]

    # Unshift
    amp_mixed = torch.fft.ifftshift(amp_mixed, dim=(-2, -1))

    # Reconstruct: mixed amplitude + original phase
    fft_mixed = amp_mixed * torch.exp(1j * phase_x)
    x_aug = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real

    return x_aug


def augment_batch(x, aug_transform, mean=0.5, std=0.5, use_fft=False,
                  fft_beta=0.5):
    """
    Apply augmentation to a normalized batch.
    Combines spatial augmentation + optional FFT amplitude swap.
    """
    # Denormalize to [0, 1]
    x_denorm = x * std + mean
    x_denorm = x_denorm.clamp(0, 1)

    # Spatial augmentation per image
    augmented = []
    for i in range(x_denorm.shape[0]):
        img_aug = aug_transform(x_denorm[i])
        augmented.append(img_aug)
    x_aug = torch.stack(augmented).to(x.device)

    # Renormalize
    x_aug = (x_aug - mean) / std

    # FFT amplitude swap (operates on normalized images)
    if use_fft:
        x_aug = fft_amplitude_swap(x_aug, beta=fft_beta)

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
                 steps=1, episodic=False, use_entropy=True,
                 use_fft=True, fft_beta=0.5):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.steps = steps
        self.episodic = episodic
        self.use_entropy = use_entropy
        self.use_fft = use_fft
        self.fft_beta = fft_beta
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
    def _get_embeddings(self, x):
        """Get raw embeddings (before L2 norm) for contrastive loss."""
        if hasattr(self.model, 'get_raw_embeddings'):
            return self.model.get_raw_embeddings(x)
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'forward_raw'):
            return self.model.backbone.forward_raw(x)
        elif hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        else:
            return self.model(x)

    def _adapt(self, x):
        # Get embeddings (single forward pass through backbone)
        z_orig = self._get_embeddings(x)

        # Augmented embeddings
        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_embeddings(x_aug)

        # Losses
        info = {}
        total_loss = torch.tensor(0.0, device=x.device)

        if self.use_entropy:
            # Get logits from embeddings (head only, no extra backbone pass)
            if hasattr(self.model, 'head'):
                logits = self.model.head(z_orig, labels=None)
            else:
                logits = self.model(x)
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

        return z_orig.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


# ══════════════════════════════════════════════════════════════
#  Feature-level Information Maximization
# ══════════════════════════════════════════════════════════════

def feature_im_loss(z, temperature=0.1):
    """
    Information Maximization at feature level (no head needed).

    Conditional entropy: each sample should have clear neighbors
      → sharpen the neighborhood distribution
    Marginal entropy: overall distribution should be uniform
      → prevent collapse to a single cluster

    Loss is clamped to non-negative: when clusters are already sharp
    (H_cond < H_marg), no gradient is produced — don't fix what works.

    z: (B, D) embeddings
    Returns: scalar loss = max(0, H_conditional - H_marginal)
    """
    z = F.normalize(z, dim=-1)
    sim = (z @ z.T) / temperature  # (B, B)

    # Mask self-similarity
    mask = torch.eye(z.shape[0], device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    p = F.softmax(sim, dim=-1)  # (B, B) neighborhood distribution

    # Conditional entropy: should be low (sharp neighborhoods)
    H_cond = -(p * (p + 1e-8).log()).sum(-1).mean()

    # Marginal entropy: should be high (uniform spread)
    p_marginal = p.mean(dim=0)
    H_marg = -(p_marginal * (p_marginal + 1e-8).log()).sum()

    return H_cond - H_marg


class ContrastiveFIM(nn.Module):
    """
    Contrastive + Feature Information Maximization.

    Loss = λ_con * NT-Xent(z, z_aug) + λ_fim * FeatureIM(z)

    NT-Xent: augmentation invariance (pull views together)
    Feature IM: sharp clusters + uniform spread (no head needed)
    FFT amplitude swap: within-domain style augmentation

    No classification head required.
    """
    def __init__(self, model, optimizer, aug_transform,
                 contrastive_lambda=1.0, contrastive_temp=0.5,
                 fim_lambda=1.0, fim_temp=0.1,
                 use_fft=True, fft_beta=0.5,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.fim_lambda = fim_lambda
        self.fim_temp = fim_temp
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps
        self.episodic = episodic
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_embeddings(self, x):
        if hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        elif hasattr(self.model, 'backbone'):
            return self.model.backbone(x)
        else:
            return self.model(x)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        # Original embeddings
        z_orig = self._get_embeddings(x)

        # Augmented embeddings (spatial + optional FFT)
        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_embeddings(x_aug)

        info = {}
        total_loss = torch.tensor(0.0, device=x.device)

        # Contrastive: pull (z_orig, z_aug) pairs together
        con_loss = nt_xent_loss(z_orig, z_aug, self.contrastive_temp)
        total_loss = total_loss + self.contrastive_lambda * con_loss
        info["contrastive"] = con_loss.item()

        # Feature IM: sharpen clusters + prevent collapse
        fim_loss = feature_im_loss(z_orig, self.fim_temp)
        total_loss = total_loss + self.fim_lambda * fim_loss
        info["fim"] = fim_loss.item()

        info["total"] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return z_orig.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)

class FIM(nn.Module):
    """
    Feature Information Maximization only.

    Loss = λ_fim * FeatureIM(z)

    No augmentation, no contrastive, no classification head.
    Just sharpen neighborhood clusters + prevent collapse.
    """
    def __init__(self, model, optimizer,
                 fim_lambda=1.0, fim_temp=0.1,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.fim_lambda = fim_lambda
        self.fim_temp = fim_temp
        self.steps = steps
        self.episodic = episodic
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_embeddings(self, x):
        if hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        elif hasattr(self.model, 'backbone'):
            return self.model.backbone(x)
        else:
            return self.model(x)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        z = self._get_embeddings(x)
        fim = feature_im_loss(z, self.fim_temp)
        total = self.fim_lambda * fim
        info = {"fim": fim.item(), "total": total.item()}
        total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return z.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)

# ══════════════════════════════════════════════════════════════
#  Nearest-Neighbor Consistency
# ══════════════════════════════════════════════════════════════

def nn_consistency_loss(z, temperature=0.1):
    """
    Nearest-neighbor consistency loss over ALL samples in batch.

    For each sample, compute similarity distribution over all other
    samples via softmax, then minimize its entropy. This pushes each
    sample to be strongly similar to some samples (same identity)
    and dissimilar to others (different identity) — creating natural
    clusters without specifying K.

    L = -Σ_i Σ_j p_ij * log(p_ij)   where p_ij = softmax(sim/τ)

    z: (B, D) embeddings
    temperature: softmax temperature (lower = sharper)
    Returns: scalar loss (mean entropy over all samples)
    """
    z = F.normalize(z, dim=-1)
    B = z.shape[0]

    # Pairwise cosine similarity
    sim = (z @ z.T) / temperature  # (B, B)

    # Mask self-similarity
    mask = torch.eye(B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Softmax → probability distribution over other samples
    p = F.softmax(sim, dim=-1)  # (B, B)

    # Entropy per sample → minimize
    H = -(p * (p + 1e-8).log()).sum(dim=-1).mean()

    return H


class ContrastiveNN(nn.Module):
    """
    Contrastive + Nearest-Neighbor Consistency.

    Loss = λ_con * NT-Xent(z, z_aug) + λ_nn * NN_consistency(z)

    NT-Xent: augmentation invariance with negatives
    NN consistency: push samples toward their K nearest neighbors
                    → tighter identity clusters without labels

    No classification head needed.
    """
    def __init__(self, model, optimizer, aug_transform,
                 contrastive_lambda=1.0, contrastive_temp=0.5,
                 nn_lambda=1.0, nn_temp=0.1,
                 use_fft=True, fft_beta=0.02,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.nn_lambda = nn_lambda
        self.nn_temp = nn_temp
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps
        self.episodic = episodic
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_embeddings(self, x):
        """Get raw embeddings (before L2 norm)."""
        if hasattr(self.model, 'get_raw_embeddings'):
            return self.model.get_raw_embeddings(x)
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'forward_raw'):
            return self.model.backbone.forward_raw(x)
        elif hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        else:
            return self.model(x)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        # Original embeddings
        z_orig = self._get_embeddings(x)

        # Augmented embeddings
        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_embeddings(x_aug)

        info = {}
        total_loss = torch.tensor(0.0, device=x.device)

        # NT-Xent contrastive (on normalized embeddings)
        con_loss = nt_xent_loss(F.normalize(z_orig, dim=-1),
                                F.normalize(z_aug, dim=-1),
                                self.contrastive_temp)
        total_loss = total_loss + self.contrastive_lambda * con_loss
        info["contrastive"] = con_loss.item()

        # NN consistency (on original embeddings)
        nn_loss = nn_consistency_loss(z_orig, temperature=self.nn_temp)
        total_loss = total_loss + self.nn_lambda * nn_loss
        info["nn"] = nn_loss.item()

        info["total"] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return z_orig.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


class ContrastivePositive(nn.Module):
    """
    Positive-only contrastive: MSE between original and augmented views.

    Loss = ||z_orig - z_aug||²

    No negatives — eliminates false negative problem where same-identity
    samples are pushed apart. Will collapse without additional regularization.
    This is a baseline to measure the false-negative impact of NT-Xent.

    No classification head needed.
    """
    def __init__(self, model, optimizer, aug_transform,
                 inv_lambda=1.0, temperature=0.5,
                 use_fft=True, fft_beta=0.02,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.inv_lambda = inv_lambda
        self.temperature = temperature
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps
        self.episodic = episodic
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_embeddings(self, x):
        """Get raw embeddings (before L2 norm)."""
        if hasattr(self.model, 'get_raw_embeddings'):
            return self.model.get_raw_embeddings(x)
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'forward_raw'):
            return self.model.backbone.forward_raw(x)
        elif hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        else:
            return self.model(x)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        z_orig = self._get_embeddings(x)

        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_embeddings(x_aug)

        # Positive-only contrastive: attraction term of NT-Xent
        # L = -sim(z, z_aug) / τ  (no negative denominator)
        z_orig_n = F.normalize(z_orig, dim=-1)
        z_aug_n = F.normalize(z_aug, dim=-1)
        pos_sim = (z_orig_n * z_aug_n).sum(dim=-1)  # (B,)
        inv_loss = -(pos_sim / self.temperature).mean()

        total = self.inv_lambda * inv_loss

        info = {
            "inv": inv_loss.item(),
            "sim": pos_sim.mean().item(),
            "total": total.item(),
        }

        total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return z_orig.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


# ══════════════════════════════════════════════════════════════
#  VICReg: Variance-Invariance-Covariance Regularization
#  Bardes et al., ICLR 2022
# ══════════════════════════════════════════════════════════════

def vicreg_variance_loss(z, gamma=1.0, eps=1e-4):
    """
    Variance term: hinge loss on per-dimension standard deviation.
    Ensures each embedding dimension has std >= gamma.
    Stops naturally when variance is high enough (no over-optimization).

    z: (B, D) embeddings
    gamma: target std threshold (default 1.0)
    Returns: scalar loss
    """
    std = torch.sqrt(z.var(dim=0) + eps)  # (D,)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(z):
    """
    Covariance term: minimize off-diagonal entries of the covariance matrix.
    Decorrelates embedding dimensions to prevent information redundancy.

    z: (B, D) embeddings
    Returns: scalar loss
    """
    B, D = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)
    # Zero out diagonal (we don't penalize self-variance)
    off_diag = cov.pow(2)
    mask = ~torch.eye(D, device=z.device).bool()
    return off_diag[mask].mean()


def vicreg_invariance_loss(z1, z2):
    """
    Invariance term: MSE between original and augmented embeddings.
    Simpler than NT-Xent — no negatives, no temperature.

    z1, z2: (B, D) embeddings (original, augmented)
    Returns: scalar loss
    """
    return F.mse_loss(z1, z2)


def vicreg_loss(z1, z2, lambda_var=1.0, lambda_inv=0.1, lambda_cov=0.04,
                gamma=1.0, inv_mode="mse", inv_temp=0.5):
    """
    Combined VICReg loss.

    z1: original embeddings (B, D) — raw, before L2 norm
    z2: augmented embeddings (B, D) — raw, before L2 norm
    inv_mode: 'mse' (original VICReg) or 'ntxent' (contrastive with negatives)

    Returns: total_loss, info dict
    """
    var_loss = (vicreg_variance_loss(z1, gamma) +
                vicreg_variance_loss(z2, gamma)) / 2

    if inv_mode == "ntxent":
        # NT-Xent on L2-normalized versions (contrastive needs unit sphere)
        inv_loss = nt_xent_loss(F.normalize(z1, dim=-1),
                                F.normalize(z2, dim=-1), inv_temp)
    else:
        inv_loss = vicreg_invariance_loss(z1, z2)

    cov_loss = (vicreg_covariance_loss(z1) +
                vicreg_covariance_loss(z2)) / 2

    total = (lambda_var * var_loss +
             lambda_inv * inv_loss +
             lambda_cov * cov_loss)

    info = {
        "var": var_loss.item(),
        "inv": inv_loss.item(),
        "cov": cov_loss.item(),
        "total": total.item(),
    }
    return total, info


def get_tta_augmentation_strong(img_size=112):
    """
    Stronger augmentation for VICReg TTA.
    Needs enough variation to make invariance term useful.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                                      ratio=(0.9, 1.1)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                    saturation=0.1, hue=0.05),
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        ], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(15),
        ], p=0.3),
    ])


class VICRegTTA(nn.Module):
    """
    VICReg-based Test-Time Adaptation.

    Uses RAW embeddings (before L2 normalization) so that variance
    and covariance terms operate on the natural scale of features.

    Loss = λ_var * Variance + λ_inv * Invariance + λ_cov * Covariance
    """
    def __init__(self, model, optimizer, aug_transform,
                 lambda_var=1.0, lambda_inv=0.1, lambda_cov=0.04,
                 gamma=1.0, inv_mode="mse", inv_temp=0.5,
                 use_fft=False, fft_beta=0.5,
                 steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.lambda_var = lambda_var
        self.lambda_inv = lambda_inv
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.inv_mode = inv_mode
        self.inv_temp = inv_temp
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps
        self.episodic = episodic
        assert steps > 0

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_raw_embeddings(self, x):
        """Get embeddings BEFORE L2 normalization for VICReg."""
        if hasattr(self.model, 'get_raw_embeddings'):
            return self.model.get_raw_embeddings(x)
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'forward_raw'):
            return self.model.backbone.forward_raw(x)
        elif hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        else:
            return self.model(x)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        z_orig = self._get_raw_embeddings(x)

        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_raw_embeddings(x_aug)

        total, info = vicreg_loss(
            z_orig, z_aug,
            lambda_var=self.lambda_var,
            lambda_inv=self.lambda_inv,
            lambda_cov=self.lambda_cov,
            gamma=self.gamma,
            inv_mode=self.inv_mode,
            inv_temp=self.inv_temp)

        total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return z_orig.detach(), info

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
