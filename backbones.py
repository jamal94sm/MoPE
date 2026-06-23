"""
backbones.py — Backbone models for TENT TTA.

Architectures:
  - ArcFace iResNet100 (CASIA-MS verification)
  - ViT-Base (ImageNet-C classification)
  - ResNet-50 / ResNet-101 (ImageNet-C classification)
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  iResNet100 (InsightFace architecture)
# ══════════════════════════════════════════════════════════════

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x);    out = self.conv1(out)
        out = self.bn2(out);  out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0.0, num_features=512,
                 groups=1, width_per_group=64):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, self.dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = self.dropout(x)
        x = x.flatten(1); x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(num_features=512, **kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3],
                   num_features=num_features, **kwargs)


# ══════════════════════════════════════════════════════════════
#  ArcFace Backbone (ONNX loading) + Classification Head
# ══════════════════════════════════════════════════════════════

class ArcFaceBackbone(nn.Module):
    """
    ArcFace iResNet100 loaded from ONNX.
    Returns L2-normalised 512-dim embeddings.
    """
    def __init__(self, onnx_path, freeze_ratio=0.0):
        super().__init__()
        import onnx
        from onnx2torch import convert

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                f"Download R100 Glint360K from InsightFace model zoo.")

        print(f"  [ArcFace] Loading ONNX: {onnx_path}")
        self.net = convert(onnx.load(onnx_path))
        print(f"  [ArcFace] Converted ONNX → PyTorch")

        if freeze_ratio > 0:
            all_params = list(self.net.parameters())
            n_freeze = int(len(all_params) * freeze_ratio)
            for i, p in enumerate(all_params):
                p.requires_grad = (i >= n_freeze)

        total = sum(p.numel() for p in self.parameters())
        print(f"  [ArcFace] Total params: {total/1e6:.2f}M")

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return F.normalize(out, p=2, dim=1)

    def forward_raw(self, x):
        """Return embeddings BEFORE L2 normalization."""
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    @property
    def embedding_dim(self):
        return 512


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head for computing logits from embeddings.
    Used during TENT adaptation to get entropy signal.

    During inference (no margin): logits = s * (emb @ W^T)
    """
    def __init__(self, num_classes, embedding_size=512, s=64.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        """
        If labels provided: ArcFace training logits (with margin).
        If labels=None: inference logits (no margin), used for TENT.
        """
        W = F.normalize(self.weight, p=2, dim=1)
        cos_theta = (embeddings @ W.t()).clamp(-1 + 1e-7, 1 - 1e-7)

        if labels is None:
            return self.s * cos_theta

        # Training: apply angular margin to target class
        sin_theta = (1.0 - cos_theta ** 2).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th,
                                  cos_theta_m,
                                  cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(
            1, labels.view(-1, 1), 1.0)
        logits = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return logits


class ArcFaceModel(nn.Module):
    """
    Combined ArcFace backbone + classification head.

    forward(x): logits without margin (for TENT entropy minimization)
    train_forward(x, labels): logits with angular margin (for ArcFace training)
    get_embeddings(x): L2-normalized 512-d embeddings (for verification)
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def get_embeddings(self, x):
        """For verification: extract L2-normalized embeddings."""
        return self.backbone(x)

    def get_raw_embeddings(self, x):
        """For VICReg: extract embeddings BEFORE L2 normalization."""
        return self.backbone.forward_raw(x)

    def forward(self, x):
        """For TENT: returns logits without margin."""
        emb = self.backbone(x)
        return self.head(emb, labels=None)

    def train_forward(self, x, labels):
        """For ArcFace training: returns logits with angular margin."""
        emb = self.backbone(x)
        return self.head(emb, labels=labels)


# ══════════════════════════════════════════════════════════════
#  Classification Backbones (ViT-Base, ResNet-50/101)
# ══════════════════════════════════════════════════════════════

def load_vit_base(num_classes=1000):
    """Load ViT-Base/16 from timm with ImageNet-1k weights."""
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    print(f"  [ViT-Base] Loaded: {model.default_cfg.get('tag', 'default')}")
    data_cfg = timm.data.resolve_model_data_config(model)
    print(f"  [ViT-Base] Normalization: mean={data_cfg['mean']}, std={data_cfg['std']}")
    return model, data_cfg


def load_resnet(variant="resnet50", num_classes=1000):
    """Load ResNet-50 or ResNet-101 from torchvision."""
    import torchvision.models as tv_models
    if variant == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    elif variant == "resnet101":
        model = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{variant}] Loaded with ImageNet-1K weights ({total/1e6:.1f}M params)")
    return model


# ══════════════════════════════════════════════════════════════
#  Unified loader
# ══════════════════════════════════════════════════════════════

def _count_casia_identities(data_dir):
    """Count unique identities (subjectID_handSide) in CASIA-MS directory."""
    identities = set()
    if not os.path.isdir(data_dir):
        return 0
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        if len(parts) >= 4:
            identities.add(f"{parts[0]}_{parts[1]}")
    return len(identities)


def build_model(cfg):
    """
    Build and return model based on cfg.backbone.

    Returns:
      For classification (imagenet_c): model that outputs logits
      For verification (casia_ms): ArcFaceModel with .get_embeddings()
                                    and forward() returning logits
    """
    if cfg.backbone == "arcface_r100":
        backbone = ArcFaceBackbone(cfg.arcface_onnx,
                                   freeze_ratio=cfg.arcface_freeze_ratio)

        # Load trained ArcFace head from checkpoint
        if cfg.arcface_ckpt and os.path.exists(cfg.arcface_ckpt):
            print(f"  [ArcFace] Loading checkpoint: {cfg.arcface_ckpt}")
            ckpt = torch.load(cfg.arcface_ckpt, map_location="cpu",
                              weights_only=False)

            # Load backbone weights
            if "model" in ckpt:
                backbone.load_state_dict(ckpt["model"], strict=False)
                print(f"  [ArcFace] Loaded backbone weights from checkpoint")

            # Determine num_classes from checkpoint
            if "arc" in ckpt:
                arc_state = ckpt["arc"]
                weight_key = [k for k in arc_state if "weight" in k][0]
                n_cls = arc_state[weight_key].shape[0]
                if cfg.arcface_num_classes is None:
                    cfg.arcface_num_classes = n_cls
                print(f"  [ArcFace] Detected {n_cls} classes from checkpoint")

            head = ArcFaceHead(cfg.arcface_num_classes,
                               embedding_size=512,
                               s=cfg.arcface_s, m=cfg.arcface_m)

            if "arc" in ckpt:
                head.load_state_dict(ckpt["arc"])
                print(f"  [ArcFace] Loaded head weights from checkpoint")
        else:
            if cfg.arcface_num_classes is None:
                # Auto-detect from dataset
                n_cls = _count_casia_identities(cfg.data_dir)
                if n_cls > 0:
                    cfg.arcface_num_classes = n_cls
                    print(f"  [ArcFace] Auto-detected {n_cls} identities "
                          f"from {cfg.data_dir}")
                else:
                    raise ValueError(
                        "ArcFace requires --arcface_ckpt (trained checkpoint) "
                        "or --arcface_num_classes. Could not auto-detect "
                        f"identities from {cfg.data_dir}")
            head = ArcFaceHead(cfg.arcface_num_classes,
                               embedding_size=512,
                               s=cfg.arcface_s, m=cfg.arcface_m)
            print(f"  [ArcFace] Initialized random head with "
                  f"{cfg.arcface_num_classes} classes (no checkpoint)")

        model = ArcFaceModel(backbone, head)
        return model.to(cfg.device)

    elif cfg.backbone == "vit_base":
        model, data_cfg = load_vit_base(cfg.num_classes)
        # Store normalization info on cfg for dataset transforms
        cfg._norm_mean = data_cfg["mean"]
        cfg._norm_std = data_cfg["std"]
        return model.to(cfg.device)

    elif cfg.backbone in ("resnet50", "resnet101"):
        model = load_resnet(cfg.backbone, cfg.num_classes)
        cfg._norm_mean = (0.485, 0.456, 0.406)
        cfg._norm_std = (0.229, 0.224, 0.225)
        return model.to(cfg.device)

    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")


# ══════════════════════════════════════════════════════════════
#  DINOv2 Backbone for JEPA-orig
# ══════════════════════════════════════════════════════════════

class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT-S/14 wrapper with patch masking support for JEPA.
    Pretrained on LVD-142M (self-supervised).
    """
    def __init__(self, model_name="dinov2_vits14", img_size=112):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.dino.embed_dim
        self.patch_size = self.dino.patch_embed.patch_size[0]
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Interpolate positional embeddings for our image size
        self._interpolate_pos_embed()

    def _interpolate_pos_embed(self):
        pos = self.dino.pos_embed.data  # (1, 1+N_orig, D)
        cls_pos = pos[:, :1, :]
        patch_pos = pos[:, 1:, :]
        orig_grid = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, -1
                                       ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos,
                                   size=(self.grid_size, self.grid_size),
                                   mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1, self.num_patches, -1)
        new_pos = torch.cat([cls_pos, patch_pos], dim=1)
        self.dino.pos_embed = nn.Parameter(new_pos, requires_grad=False)

    def _prepare_tokens(self, x):
        B = x.shape[0]
        patches = self.dino.patch_embed(x)  # (B, N, D)
        cls = self.dino.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B, 1+N, D)
        tokens = tokens + self.dino.pos_embed
        return tokens

    def forward_patches(self, x, patch_mask=None):
        """
        Forward with optional patch masking for JEPA.

        patch_mask: (B, M) integer indices of visible patches (0 to N-1)
        Returns: (B, 1+M, D) if masked, (B, 1+N, D) if not
        """
        tokens = self._prepare_tokens(x)

        if patch_mask is not None:
            B, M = patch_mask.shape
            cls = tokens[:, :1, :]
            patches = tokens[:, 1:, :]
            idx = patch_mask.unsqueeze(-1).expand(B, M, self.embed_dim)
            selected = torch.gather(patches, 1, idx)
            tokens = torch.cat([cls, selected], dim=1)

        for blk in self.dino.blocks:
            tokens = blk(tokens)
        tokens = self.dino.norm(tokens)
        return tokens

    def forward(self, x):
        """CLS token, L2 normalized."""
        tokens = self.forward_patches(x)
        return F.normalize(tokens[:, 0, :], dim=-1)

    def forward_raw(self, x):
        """CLS token, raw (no L2 norm)."""
        tokens = self.forward_patches(x)
        return tokens[:, 0, :]

    def freeze_except_last_n(self, n_blocks):
        """Freeze all except last n transformer blocks + final norm."""
        self.requires_grad_(False)
        total = len(self.dino.blocks)
        if n_blocks > 0:
            for blk in self.dino.blocks[total - n_blocks:]:
                blk.requires_grad_(True)
        self.dino.norm.requires_grad_(True)

    def configure_for_tta(self, n_blocks=0):
        """Freeze everything, unfreeze all LayerNorm + last n blocks."""
        self.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
        total = len(self.dino.blocks)
        if n_blocks > 0:
            for blk in self.dino.blocks[total - n_blocks:]:
                blk.requires_grad_(True)


class DINOv2Model(nn.Module):
    """Wrapper for DINOv2 compatible with evaluation pipeline."""
    def __init__(self, model_name="dinov2_vits14", img_size=112):
        super().__init__()
        self.backbone = DINOv2Backbone(model_name, img_size)
        self.head = None

    def get_embeddings(self, x):
        return self.backbone(x)

    def get_raw_embeddings(self, x):
        return self.backbone.forward_raw(x)

    def forward_patches(self, x, patch_mask=None):
        return self.backbone.forward_patches(x, patch_mask)

    def forward(self, x):
        return self.backbone(x)
