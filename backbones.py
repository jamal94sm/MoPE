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

    forward(x) returns logits (for TENT — model(x) must return logits)
    get_embeddings(x) returns L2-normalized 512-d embeddings (for verification)
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def get_embeddings(self, x):
        """For verification: extract L2-normalized embeddings."""
        return self.backbone(x)

    def forward(self, x):
        """For TENT: returns logits (no margin, inference mode)."""
        emb = self.backbone(x)
        return self.head(emb, labels=None)


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

def build_model(cfg):
    """
    Build and return model based on cfg.backbone.

    Returns:
      For classification (imagenet_c): model that outputs logits
      For verification (casia_ms): ArcFaceModel with .get_embeddings() and .get_logits()
    """
    if cfg.backbone == "arcface_r100":
        backbone = ArcFaceBackbone(cfg.arcface_onnx)

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
                raise ValueError(
                    "ArcFace requires --arcface_ckpt (trained checkpoint) "
                    "or --arcface_num_classes for TENT. The classification "
                    "head is needed for entropy computation.")
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
