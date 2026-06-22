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

        self.loss_fn = (F.smooth_l1_loss if loss_fn == "smooth_l1"
                        else F.mse_loss)

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

        self.jepa_loss_fn = (F.smooth_l1_loss if loss_fn == "smooth_l1"
                             else F.mse_loss)

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
