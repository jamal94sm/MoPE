"""
tent.py — TTA methods: Contrastive (NT-Xent) and JEPA.

Contrastive: NT-Xent on augmented views with negatives
JEPA: BYOL-style prediction with EMA teacher (positive-only, no collapse)

Both operate on raw embeddings (before L2 norm), update only BN affine params.
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ══════════════════════════════════════════════════════════════
#  BN Configuration
# ══════════════════════════════════════════════════════════════

def configure_model(model):
    """Original TENT: train mode, null running stats, BN affine only."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def configure_model_safe(model):
    """Safe BN: preserve running stats, update via EMA in train mode."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
    return model


def collect_params(model):
    """Collect BN affine parameters (weight + bias)."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for pn, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


# ══════════════════════════════════════════════════════════════
#  Augmentation
# ══════════════════════════════════════════════════════════════

def get_tta_augmentation(img_size=112):
    """Strong augmentation for TTA."""
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


def fft_amplitude_swap(x, beta=0.02):
    """FFT augmentation: swap low-freq amplitude with random batch partner."""
    B, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    x_partner = x[perm]

    fft_x = torch.fft.fft2(x, dim=(-2, -1))
    fft_p = torch.fft.fft2(x_partner, dim=(-2, -1))

    amp_x = torch.abs(fft_x)
    phase_x = torch.angle(fft_x)
    amp_p = torch.abs(fft_p)

    cy, cx = H // 2, W // 2
    rh, rw = int(H * beta * 0.5), int(W * beta * 0.5)

    amp_x_s = torch.fft.fftshift(amp_x, dim=(-2, -1))
    amp_p_s = torch.fft.fftshift(amp_p, dim=(-2, -1))

    amp_mix = amp_x_s.clone()
    y1, y2 = max(0, cy - rh), min(H, cy + rh)
    x1, x2 = max(0, cx - rw), min(W, cx + rw)
    amp_mix[:, :, y1:y2, x1:x2] = amp_p_s[:, :, y1:y2, x1:x2]

    amp_mix = torch.fft.ifftshift(amp_mix, dim=(-2, -1))
    fft_mix = amp_mix * torch.exp(1j * phase_x)
    return torch.fft.ifft2(fft_mix, dim=(-2, -1)).real


def augment_batch(x, aug_transform, mean=0.5, std=0.5,
                  use_fft=False, fft_beta=0.02):
    """Apply spatial augmentation + optional FFT swap."""
    x_denorm = (x * std + mean).clamp(0, 1)
    augmented = [aug_transform(x_denorm[i]) for i in range(x_denorm.shape[0])]
    x_aug = torch.stack(augmented).to(x.device)
    x_aug = (x_aug - mean) / std
    if use_fft:
        x_aug = fft_amplitude_swap(x_aug, beta=fft_beta)
    return x_aug


# ══════════════════════════════════════════════════════════════
#  NT-Xent Contrastive Loss
# ══════════════════════════════════════════════════════════════

def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent contrastive loss with negatives."""
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


class Contrastive(nn.Module):
    """NT-Xent contrastive TTA on raw embeddings. No head needed."""

    def __init__(self, model, optimizer, aug_transform,
                 contrastive_lambda=1.0, contrastive_temp=0.5,
                 use_fft=True, fft_beta=0.02, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def _get_embeddings(self, x):
        if hasattr(self.model, 'get_raw_embeddings'):
            return self.model.get_raw_embeddings(x)
        elif hasattr(self.model, 'backbone') and \
                hasattr(self.model.backbone, 'forward_raw'):
            return self.model.backbone.forward_raw(x)
        elif hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(x)
        else:
            return self.model(x)

    def forward(self, x):
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        z_orig = self._get_embeddings(x)
        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_embeddings(x_aug)

        con_loss = nt_xent_loss(F.normalize(z_orig, dim=-1),
                                F.normalize(z_aug, dim=-1),
                                self.contrastive_temp)
        total = self.contrastive_lambda * con_loss

        info = {"con": con_loss.item(), "total": total.item()}

        total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return z_orig.detach(), info

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)


# ══════════════════════════════════════════════════════════════
#  JEPA-TTA: Predictive Architecture with EMA Teacher
# ══════════════════════════════════════════════════════════════

class PredictorMLP(nn.Module):
    """Bottleneck MLP: prevents trivial identity mapping."""
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def ema_update(student, teacher, momentum):
    """Update teacher as EMA of student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


class JEPATTA(nn.Module):
    """
    JEPA-style TTA: positive-only prediction with EMA teacher.

    Student(x) → z_s → Predictor → z_p
    Teacher(aug(x)) → z_t  (stop gradient)
    Loss = ||z_p - z_t||

    No collapse because:
    - Teacher is slowly moving (EMA ~0.996)
    - Predictor bottleneck (512→256→512)
    - Asymmetry: student sees original, teacher sees augmented

    No classification head needed.
    """
    def __init__(self, model, optimizer, aug_transform, predictor,
                 momentum=0.996, loss_fn="smooth_l1",
                 use_fft=True, fft_beta=0.02, steps=1):
        super().__init__()
        self.student = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.predictor = predictor
        self.momentum = momentum
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps

        if loss_fn == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif loss_fn == "cosine":
            self.loss_fn = lambda z_p, z_t: (
                1 - F.cosine_similarity(z_p, z_t, dim=-1)).mean()
        else:
            self.loss_fn = F.mse_loss

        # Teacher = frozen EMA copy of student
        self.teacher = deepcopy(model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        # Save for episodic reset
        self.student_state = deepcopy(self.student.state_dict())
        self.teacher_state = deepcopy(self.teacher.state_dict())
        self.predictor_state = deepcopy(self.predictor.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def _get_raw(self, model, x):
        if hasattr(model, 'get_raw_embeddings'):
            return model.get_raw_embeddings(x)
        elif hasattr(model, 'backbone') and \
                hasattr(model.backbone, 'forward_raw'):
            return model.backbone.forward_raw(x)
        elif hasattr(model, 'get_embeddings'):
            return model.get_embeddings(x)
        else:
            return model(x)

    def forward(self, x):
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        # Student: original image
        z_s = self._get_raw(self.student, x)

        # Teacher: augmented image (detached)
        with torch.no_grad():
            x_aug = augment_batch(x, self.aug_transform,
                                  use_fft=self.use_fft,
                                  fft_beta=self.fft_beta)
            z_t = self._get_raw(self.teacher, x_aug)

        # Predict teacher from student
        z_p = self.predictor(z_s)
        loss = self.loss_fn(z_p, z_t)

        # Monitor
        with torch.no_grad():
            sim = F.cosine_similarity(z_s, z_t, dim=-1).mean().item()
            p_std = z_p.std(dim=0).mean().item()
            t_std = z_t.std(dim=0).mean().item()

        info = {
            "loss": loss.item(),
            "sim": sim,
            "p_std": p_std,
            "t_std": t_std,
            "total": loss.item(),
        }

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EMA update
        with torch.no_grad():
            ema_update(self.student, self.teacher, self.momentum)

        return z_s.detach(), info

    def reset(self):
        self.student.load_state_dict(
            {k: v.clone() for k, v in self.student_state.items()})
        self.teacher.load_state_dict(
            {k: v.clone() for k, v in self.teacher_state.items()})
        self.predictor.load_state_dict(
            {k: v.clone() for k, v in self.predictor_state.items()})
        self.optimizer.load_state_dict(self.optimizer_state)


class JEPAContrastive(nn.Module):
    """
    Combined NT-Xent + JEPA at TTA time.

    L = λ_con × NT-Xent(z, z_aug) + λ_jepa × smooth_l1(pred(z), z_teacher)

    NT-Xent: immediate domain adaptation via contrastive signal
    JEPA: stability via prediction consistency with EMA teacher
    """
    def __init__(self, model, optimizer, aug_transform, predictor,
                 con_lambda=1.0, con_temp=0.5,
                 jepa_lambda=1.0, momentum=0.996,
                 loss_fn="smooth_l1",
                 use_fft=True, fft_beta=0.02, steps=1):
        super().__init__()
        self.student = model
        self.optimizer = optimizer
        self.aug_transform = aug_transform
        self.predictor = predictor
        self.con_lambda = con_lambda
        self.con_temp = con_temp
        self.jepa_lambda = jepa_lambda
        self.momentum = momentum
        self.use_fft = use_fft
        self.fft_beta = fft_beta
        self.steps = steps

        if loss_fn == "smooth_l1":
            self.jepa_loss_fn = F.smooth_l1_loss
        elif loss_fn == "cosine":
            self.jepa_loss_fn = lambda z_p, z_t: (
                1 - F.cosine_similarity(z_p, z_t, dim=-1)).mean()
        else:
            self.jepa_loss_fn = F.mse_loss

        self.teacher = deepcopy(model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.student_state = deepcopy(self.student.state_dict())
        self.teacher_state = deepcopy(self.teacher.state_dict())
        self.predictor_state = deepcopy(self.predictor.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def _get_raw(self, model, x):
        if hasattr(model, 'get_raw_embeddings'):
            return model.get_raw_embeddings(x)
        elif hasattr(model, 'backbone') and \
                hasattr(model.backbone, 'forward_raw'):
            return model.backbone.forward_raw(x)
        elif hasattr(model, 'get_embeddings'):
            return model.get_embeddings(x)
        else:
            return model(x)

    def forward(self, x):
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        # Student: original
        z_s = self._get_raw(self.student, x)

        # Augmented
        x_aug = augment_batch(x, self.aug_transform,
                              use_fft=self.use_fft, fft_beta=self.fft_beta)
        z_aug = self._get_raw(self.student, x_aug)

        # Teacher: augmented (detached)
        with torch.no_grad():
            z_t = self._get_raw(self.teacher, x_aug)

        total = torch.tensor(0.0, device=x.device)
        info = {}

        # NT-Xent contrastive
        con_loss = nt_xent_loss(F.normalize(z_s, dim=-1),
                                F.normalize(z_aug, dim=-1),
                                self.con_temp)
        total = total + self.con_lambda * con_loss
        info["con"] = con_loss.item()

        # JEPA prediction
        z_p = self.predictor(z_s)
        jepa_loss = self.jepa_loss_fn(z_p, z_t)
        total = total + self.jepa_lambda * jepa_loss
        info["jepa"] = jepa_loss.item()

        info["total"] = total.item()

        with torch.no_grad():
            info["sim"] = F.cosine_similarity(z_s, z_t, dim=-1).mean().item()

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        with torch.no_grad():
            ema_update(self.student, self.teacher, self.momentum)

        return z_s.detach(), info

    def reset(self):
        self.student.load_state_dict(
            {k: v.clone() for k, v in self.student_state.items()})
        self.teacher.load_state_dict(
            {k: v.clone() for k, v in self.teacher_state.items()})
        self.predictor.load_state_dict(
            {k: v.clone() for k, v in self.predictor_state.items()})
        self.optimizer.load_state_dict(self.optimizer_state)


# ══════════════════════════════════════════════════════════════
#  JEPA-Original: Patch masking + Transformer predictor
#  Adapted from I-JEPA (Assran et al., 2023)
# ══════════════════════════════════════════════════════════════

def patchify(batch_size, num_patches, num_blocks=4,
             trg_ratio=(0.15, 0.20), ctx_ratio=(0.85, 1.00),
             ar_range=(0.75, 1.5), device="cpu"):
    """
    Create context and target masks for JEPA.
    Returns: [ctx_mask], [tgt_mask1, ..., tgt_maskK]
      ctx_mask: (B, N_ctx) integer indices of visible patches
      tgt_masks: each (B, N_tgt) integer indices of target patches
    """
    import math
    H = W = int(num_patches ** 0.5)
    P = H * W

    def sample_block(scale):
        s = torch.empty(()).uniform_(*scale).item()
        ar = torch.empty(()).uniform_(*ar_range).item()
        area = max(1, int(s * P))
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(area / h))))
        y = torch.randint(0, max(1, H - h + 1), ())
        x = torch.randint(0, max(1, W - w + 1), ())
        idx = [(y+i)*W + (x+j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device)

    ctx_masks = []
    tgt_masks = [[] for _ in range(num_blocks)]
    min_ctx = P
    min_tgt = P

    for _ in range(batch_size):
        occupied = torch.zeros(P, dtype=torch.bool, device=device)
        for k in range(num_blocks):
            idx = sample_block(trg_ratio)
            tgt_masks[k].append(idx)
            occupied[idx] = True
            min_tgt = min(min_tgt, idx.numel())
        for _ in range(10):
            ctx = sample_block(ctx_ratio)
            ctx = ctx[~occupied[ctx]]
            if ctx.numel() > 0:
                break
        else:
            ctx = (~occupied).nonzero().squeeze(1)
        min_ctx = min(min_ctx, ctx.numel())
        ctx_masks.append(ctx)

    ctx_out = torch.stack([
        c[torch.randperm(c.numel(), device=device)[:min_ctx]]
        for c in ctx_masks
    ])
    tgt_out = [
        torch.stack([
            t[torch.randperm(t.numel(), device=device)[:min_tgt]]
            for t in tgt_masks[k]
        ])
        for k in range(num_blocks)
    ]
    return [ctx_out], tgt_out


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2D sinusoidal positional embeddings."""
    import numpy as np
    def _1d(dim, pos):
        omega = np.arange(dim // 2, dtype=float) / (dim / 2.)
        omega = 1. / (10000 ** omega)
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    gh = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(gh, gh)
    emb_h = _1d(embed_dim // 2, grid[0].flatten())
    emb_w = _1d(embed_dim // 2, grid[1].flatten())
    return np.concatenate([emb_h, emb_w], axis=1)


def _gather(x, mask):
    """Gather patches by index. x:(B,P,D), mask:(B,N) → (B,N,D)"""
    B, N = mask.shape
    D = x.size(-1)
    return torch.gather(x, 1, mask.unsqueeze(-1).expand(B, N, D))


class JEPAPredictor(nn.Module):
    """
    Transformer predictor for JEPA.
    Takes context patch embeddings + mask tokens → predicts target patches.
    """
    def __init__(self, num_patches, embed_dim, pred_dim=None, depth=6):
        super().__init__()
        if pred_dim is None:
            pred_dim = embed_dim // 2
        num_heads = max(1, pred_dim // 64)
        while pred_dim % num_heads != 0:
            num_heads -= 1

        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

        pos = get_2d_sincos_pos_embed(pred_dim,
                                       int(num_patches ** 0.5))
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=num_heads,
            dim_feedforward=int(pred_dim * 4),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(pred_dim)

    def forward(self, context, context_masks, target_masks):
        """
        context:       (B*n_ctx, N_ctx, D)
        context_masks: list[(B, N_ctx)]
        target_masks:  list[(B, N_tgt)]
        Returns:       (B*n_ctx*n_tgt, N_tgt, D)
        """
        if not isinstance(context_masks, list):
            context_masks = [context_masks]
        if not isinstance(target_masks, list):
            target_masks = [target_masks]

        n_ctx = len(context_masks)
        n_tgt = len(target_masks)
        B = context.size(0) // n_ctx
        N_tgt = target_masks[0].size(1)

        x = self.in_proj(context)

        pos_full = self.pos_embed.expand(B, -1, -1)
        pos_ctx = torch.cat(
            [_gather(pos_full, m) for m in context_masks], dim=0)
        x = x + pos_ctx

        pos_tgt = torch.cat(
            [_gather(pos_full, m) for m in target_masks], dim=0)
        mask_tokens = self.mask_token.expand(
            pos_tgt.size(0), N_tgt, -1) + pos_tgt

        x = x.repeat(n_tgt, 1, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x)

        preds = x[:, -N_tgt:]
        return self.out_proj(preds)


def _repeat_interleave_batch(x, B, repeat):
    """Tile x so each group of B rows is repeated."""
    N, D = x.size(1), x.size(2)
    num_blocks = x.size(0) // B
    x = x.view(B, num_blocks, N, D)
    x = x.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
    return x.reshape(B * repeat * num_blocks, N, D)


def apply_masks(x, masks):
    """Gather patches for each mask. x:(B,P,D), masks:list[(B,N)]"""
    out = []
    for m in masks:
        out.append(_gather(x, m))
    return torch.cat(out, dim=0)


def collect_ln_params(model):
    """Collect all LayerNorm parameters (for DINOv2 TTA)."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for pn, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


class JEPAOriginal(nn.Module):
    """
    Original JEPA with patch masking + Transformer predictor.

    Context encoder sees ~85% patches → embeddings
    Target encoder (EMA) sees 100% patches → target embeddings
    Predictor: context embeddings + mask tokens → predict target patches

    Loss = smooth_l1(predicted_patches, target_patches)

    For TTA: only LayerNorm params updated in context encoder.
    Predictor stays frozen (already learned from source).
    """
    def __init__(self, context_encoder, target_encoder, predictor,
                 optimizer, num_patches, num_blocks=4,
                 trg_ratio=(0.15, 0.20), ctx_ratio=(0.85, 1.00),
                 momentum=0.996, loss_fn="smooth_l1",
                 update_predictor=True, steps=1):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.optimizer = optimizer
        self.num_patches = num_patches
        self.num_blocks = num_blocks
        self.trg_ratio = tuple(trg_ratio)
        self.ctx_ratio = tuple(ctx_ratio)
        self.momentum = momentum
        self.update_predictor = update_predictor
        self.steps = steps

        if loss_fn == "cosine":
            self.loss_fn = lambda p, t: (
                1 - F.cosine_similarity(
                    p.reshape(-1, p.size(-1)),
                    t.reshape(-1, t.size(-1)), dim=-1)).mean()
        elif loss_fn == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            self.loss_fn = F.mse_loss

        self.ctx_state = deepcopy(context_encoder.state_dict())
        self.tgt_state = deepcopy(target_encoder.state_dict())
        self.pred_state = deepcopy(predictor.state_dict())
        self.opt_state = deepcopy(optimizer.state_dict())

    def forward(self, x):
        for _ in range(self.steps):
            outputs, info = self._adapt(x)
        return outputs, info

    @torch.enable_grad()
    def _adapt(self, x):
        B = x.shape[0]
        device = x.device

        ctx_masks, tgt_masks = patchify(
            B, self.num_patches, self.num_blocks,
            trg_ratio=self.trg_ratio, ctx_ratio=self.ctx_ratio,
            device=device)

        # Context encoder: masked patches
        ctx_embeds = self.context_encoder.forward_patches(
            x, ctx_masks[0])[:, 1:, :]  # remove CLS

        # Target encoder: all patches (no grad)
        with torch.no_grad():
            tgt_full = self.target_encoder.forward_patches(x)
            tgt_full = tgt_full[:, 1:, :]  # remove CLS
            tgt_embeds = apply_masks(tgt_full, tgt_masks)
            tgt_embeds = _repeat_interleave_batch(
                tgt_embeds, B, repeat=len(ctx_masks))

        # Predictor
        if self.update_predictor:
            pred_embeds = self.predictor(ctx_embeds, ctx_masks, tgt_masks)
        else:
            with torch.no_grad():
                pred_embeds = self.predictor(
                    ctx_embeds, ctx_masks, tgt_masks)
            # Re-enable grad for encoder params by recomputing
            pred_embeds = self.predictor(ctx_embeds, ctx_masks, tgt_masks)

        loss = self.loss_fn(pred_embeds, tgt_embeds)

        with torch.no_grad():
            sim = F.cosine_similarity(
                pred_embeds.reshape(-1, pred_embeds.size(-1)),
                tgt_embeds.reshape(-1, tgt_embeds.size(-1)),
                dim=-1).mean().item()
            p_std = pred_embeds.std(dim=-2).mean().item()

        info = {
            "loss": loss.item(),
            "sim": sim,
            "p_std": p_std,
            "total": loss.item(),
        }

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            ema_update(self.context_encoder, self.target_encoder,
                       self.momentum)

        return ctx_embeds[:B, 0, :].detach(), info

    def reset(self):
        self.context_encoder.load_state_dict(
            {k: v.clone() for k, v in self.ctx_state.items()})
        self.target_encoder.load_state_dict(
            {k: v.clone() for k, v in self.tgt_state.items()})
        self.predictor.load_state_dict(
            {k: v.clone() for k, v in self.pred_state.items()})
        self.optimizer.load_state_dict(self.opt_state)
