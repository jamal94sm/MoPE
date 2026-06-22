"""
config.py — Configuration for Test-Time Adaptation.

Modes: contrastive (NT-Xent), jepa (BYOL-style prediction)
Datasets: ImageNet-C (classification), CASIA-MS (verification)
"""

import argparse

IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CASIA_MS_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]

CASIA_ORACLE_DOMAINS = {
    "visible": ["WHT", "460"],
    "red_nir": ["630", "700"],
    "nir":     ["850", "940"],
}

CASIA_ORACLE_LOOKUP = {}
for _gid, (_gname, _spectrums) in enumerate(CASIA_ORACLE_DOMAINS.items()):
    for _s in _spectrums:
        CASIA_ORACLE_LOOKUP[_s] = (_gname, _gid)


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="Test-Time Adaptation")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--dataset", default="imagenet_c",
                   choices=["imagenet_c", "casia_ms"])
    p.add_argument("--data_dir", default="./data/ImageNet-C")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # ─── CASIA-MS ─────────────────────────────────────────────
    p.add_argument("--train_spectrums", nargs="*", default=["WHT"])
    p.add_argument("--test_id_ratio", type=float, default=0.5)
    p.add_argument("--gallery_ratio", type=float, default=0.1)
    p.add_argument("--oracle_domains", action="store_true", default=False)

    # ─── ArcFace training ─────────────────────────────────────
    p.add_argument("--arcface_epochs", type=int, default=20)
    p.add_argument("--arcface_head_epochs", type=int, default=10)
    p.add_argument("--arcface_lr", type=float, default=1e-4)
    p.add_argument("--arcface_lr_phase2", type=float, default=1e-2)
    p.add_argument("--arcface_wd", type=float, default=5e-4)
    p.add_argument("--arcface_eval_every", type=int, default=5)
    p.add_argument("--arcface_freeze_ratio", type=float, default=0.75)
    p.add_argument("--arcface_m", type=float, default=0.50)
    p.add_argument("--arcface_m_phase2", type=float, default=0.10)
    p.add_argument("--arcface_s", type=float, default=64.0)

    # ─── Backbone ─────────────────────────────────────────────
    p.add_argument("--backbone", default="vit_base",
                   choices=["vit_base", "resnet50", "resnet101",
                            "arcface_r100"])
    p.add_argument("--arcface_onnx", type=str,
                   default="/home/pai-ng/Jamal/NIPS2026/face_models/"
                           "checkpoints/r100_glint360k.onnx")
    p.add_argument("--arcface_ckpt", type=str, default=None)
    p.add_argument("--arcface_num_classes", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--img_size", type=int, default=224)

    # ─── TTA method ───────────────────────────────────────────
    p.add_argument("--tta_method", default="contrastive",
                   choices=["contrastive", "jepa"],
                   help="'contrastive' = NT-Xent (no head), "
                        "'jepa' = BYOL-style prediction (no head)")
    p.add_argument("--tent_lr", type=float, default=1e-4)
    p.add_argument("--tent_steps", type=int, default=1)
    p.add_argument("--reset_tta", action="store_true", default=False,
                   help="Episodic: reset before each domain")
    p.add_argument("--no_reset_tta", dest="reset_tta",
                   action="store_false")
    p.add_argument("--safe_bn", action="store_true", default=True)
    p.add_argument("--no_safe_bn", dest="safe_bn", action="store_false")

    # ─── Contrastive params ───────────────────────────────────
    p.add_argument("--contrastive_lambda", type=float, default=1.0)
    p.add_argument("--contrastive_temp", type=float, default=0.5)

    # ─── JEPA params ──────────────────────────────────────────
    p.add_argument("--jepa_momentum", type=float, default=0.996,
                   help="EMA momentum for teacher (0.99-0.999)")
    p.add_argument("--jepa_pred_dim", type=int, default=256,
                   help="Predictor MLP hidden dimension")
    p.add_argument("--jepa_loss", default="smooth_l1",
                   choices=["smooth_l1", "mse"])
    p.add_argument("--jepa_warmup_epochs", type=int, default=0,
                   help="Epochs to warm up predictor on source (Phase 1.5)")
    p.add_argument("--jepa_train_lambda", type=float, default=1.0,
                   help="Weight of JEPA loss during Phase 1 joint training")

    # ─── Augmentation ─────────────────────────────────────────
    p.add_argument("--use_fft_aug", action="store_true", default=True)
    p.add_argument("--no_fft_aug", dest="use_fft_aug",
                   action="store_false")
    p.add_argument("--fft_beta", type=float, default=0.02)

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_tent")

    cfg = p.parse_args(args)

    if cfg.dataset == "casia_ms":
        cfg.is_verification = True
        if cfg.backbone != "arcface_r100":
            print(f"[WARN] CASIA-MS requires arcface_r100, "
                  f"overriding '{cfg.backbone}'")
            cfg.backbone = "arcface_r100"
        cfg.img_size = 112
    else:
        cfg.is_verification = False

    return cfg
