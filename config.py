"""
config.py — All hyperparameters from AAAI 2026 paper (Appendix F).
"""

import argparse, math

# ─── Corruption sequences ────────────────────────────────────────────
IMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CIFAR100_C_CORRUPTIONS = IMAGENET_C_CORRUPTIONS          # same 15 types

CRS_DOMAINS = ["imagenet_v2", "imagenet_a", "imagenet_r", "imagenet_sketch"]

ACDC_DOMAINS = ["fog", "night", "rain", "snow"]

# Oracle domain families for controlled cross-expert experiments.
# One expert per group. Within-group experts should help each other,
# across-group experts should not.
ORACLE_DOMAINS = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

# Reverse lookup: corruption_name → (group_name, group_id)
ORACLE_LOOKUP = {}
for gid, (gname, corruptions) in enumerate(ORACLE_DOMAINS.items()):
    for c in corruptions:
        ORACLE_LOOKUP[c] = (gname, gid)

# ─── Class mappings for partial-class datasets ────────────────────────
# ImageNet-A and ImageNet-R use 200 of the 1000 ImageNet classes.
# These mappings are loaded at runtime from the dataset folders.

# ─── Defaults ────────────────────────────────────────────────────────
def get_cfg(args=None):
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", default="imagenet_c",
                   choices=["imagenet_c", "cifar100_c",
                            "imagenet_plus", "imagenet_plusplus", "acdc"])
    p.add_argument("--data_dir", default="./data/ImageNet-C")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", nargs="*", default=None,
                   help="Corruption(s) to run. None=all 15. "
                        "Examples: --corruptions glass_blur | "
                        "--corruptions glass_blur defocus_blur fog")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_rounds", type=int, default=3,
                   help="Rounds for CRS benchmark")

    # backbone
    # vit_base_patch16_224.orig_in21k_ft_in1k, vit_base_patch16_224, vit_base_patch16_224.augreg_in21k_ft_in1k
    p.add_argument("--backbone", default="vit_base_patch16_224")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=1000)

    # expert architecture (Section 3.2)
    p.add_argument("--shared_rank", type=int, default=32,
                   help="LoRA rank for shared expert branch")
    p.add_argument("--domain_rank", type=int, default=16,
                   help="LoRA rank for each domain self-adaptive expert")
    p.add_argument("--num_experts_per_moe", type=int, default=2,
                   help="M: number of experts inside each MoE module")
    p.add_argument("--fusion_lambda", type=float, default=0.5,
                   help="λ: balance shared vs domain branch (Eq.5)")
    p.add_argument("--use_shared_expert", action="store_true", default=True,
                   help="Use shared expert branch. If False, only domain experts.")
    p.add_argument("--no_shared_expert", dest="use_shared_expert",
                   action="store_false")

    # ─── ViDA-style high-rank domain experts ───
    # Standard: both shared (rank 32) and domain (rank 16) are low-rank.
    # ViDA mode: shared stays low-rank, domain experts use high-rank
    # adapters with much larger bottleneck for greater capacity.
    # Rationale: domain-specific features need more capacity to capture
    # fine-grained corruption patterns, while shared features compress well.
    p.add_argument("--vida_domain", action="store_true", default=False,
                   help="Use high-rank domain experts (ViDA-style). "
                        "Shared expert stays low-rank (shared_rank). "
                        "Domain experts use domain_high_rank instead of domain_rank.")
    p.add_argument("--no_vida_domain", dest="vida_domain", action="store_false")
    p.add_argument("--domain_high_rank", type=int, default=128,
                   help="Bottleneck dim for high-rank domain experts when "
                        "--vida_domain is set. Higher = more capacity. "
                        "Recommended: 64-256. Paper ViDA uses 256.")

    # FDD (Section 3.3)
    p.add_argument("--fdd_freq_radius", type=int, default=16,
                   help="l: frequency radius for low-freq crop")
    p.add_argument("--fdd_threshold", type=float, default=1.5,
                   help="τ: Mahalanobis threshold for new domain")
    p.add_argument("--fdd_shrinkage", type=float, default=0.1,
                   help="ε: covariance shrinkage")
    p.add_argument("--fdd_init_var", type=float, default=1.0,
                   help="σ²₀: initial diagonal covariance for new domain")
    p.add_argument("--fdd_diagonal", action="store_true", default=True,
                   help="Approximate covariance as diagonal (saves ~60% GPU mem)")

    # optimisation (Section 3.4, Appendix F)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (auto-set per dataset if None)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--confidence_threshold", type=float, default=0.4,
                   help="κ coefficient: threshold = κ × ln(C)")

    # anti-collapse fixes
    p.add_argument("--entropy_floor", type=float, default=0.0,
                   help="Skip update when entropy < this value "
                        "(prevents reinforcing overconfident predictions). "
                        "0 = disabled. Recommended: 0.05")
    p.add_argument("--stochastic_restore", type=float, default=0.0,
                   help="Probability of resetting each shared expert param "
                        "to its initial value each step (CoTTA-style). "
                        "0 = disabled. Recommended: 0.01")
    p.add_argument("--div_lambda", type=float, default=0.5,
                   help="Batch diversity regularizer weight (IM loss). "
                        "Maximizes entropy of batch-mean prediction to prevent "
                        "single-class collapse. 0 = disabled. Recommended: 1.0")

    # ─── Cross-expert pseudo-label supervision ───
    # Experts help each other learn via consensus pseudo-labels and
    # knowledge distillation from backbone + previous experts.
    #
    # For domain k, the "teacher ensemble" consists of:
    #   - Frozen backbone (always available)
    #   - Expert 0 through k-1 (each with the shared expert at their time)
    #
    # Pseudo-label supervision: backbone + previous experts generate
    # labels for the current expert via group consensus.
    #
    # Two modes:
    #   Hard PL: cross-entropy with argmax class (one-hot target)
    #   Soft PL: cross-entropy with teacher's full distribution
    #            sharpened by temperature (lower T = sharper)
    #
    # Total loss = ent_loss + pl_lambda * pl_loss + div_lambda * div_loss

    p.add_argument("--use_pseudo_labels", action="store_true", default=False,
                   help="Enable cross-expert pseudo-label supervision")
    p.add_argument("--no_pseudo_labels", dest="use_pseudo_labels",
                   action="store_false")

    p.add_argument("--pl_lambda", type=float, default=0.5,
                   help="Weight for pseudo-label loss. "
                        "Recommended: 0.3-1.0")

    p.add_argument("--pl_threshold", type=float, default=0.9,
                   help="Confidence threshold for pseudo-label acceptance. "
                        "A voter's prediction counts only if max(softmax) > this. "
                        "Recommended: 0.85-0.95")

    p.add_argument("--pl_agreement", type=float, default=0.8,
                   help="Fraction of voters that must agree on the same class. "
                        "1.0 = unanimous agreement required. "
                        "0.5 = simple majority. "
                        "Recommended: 0.8 (80%% of voters)")

    p.add_argument("--pl_soft", action="store_true", default=False,
                   help="Use soft pseudo-labels (teacher distribution) instead "
                        "of hard (argmax one-hot). Soft PL transfers inter-class "
                        "relationships but gives weaker directional signal.")
    p.add_argument("--pl_hard", dest="pl_soft", action="store_false",
                   help="Use hard pseudo-labels (default)")

    p.add_argument("--pl_sharpness", type=float, default=0.5,
                   help="Temperature for soft pseudo-labels. Controls how peaked "
                        "the teacher distribution is. Lower = sharper (closer to "
                        "hard labels). Higher = softer (more inter-class info). "
                        "0.1 = nearly hard. 1.0 = standard softmax. 2.0 = very soft. "
                        "Only used when --pl_soft is set. Recommended: 0.3-1.0")

    p.add_argument("--pl_warmup", type=int, default=100,
                   help="Number of batches after a new domain is detected before "
                        "PL/KD losses are applied. During warmup, only entropy + "
                        "diversity losses update the expert. This lets the expert "
                        "learn basic domain features before receiving cross-expert "
                        "supervision. 0 = no warmup. Recommended: 30-100")

    # ─── Teacher selection for PL ───
    p.add_argument("--fdd_include_threshold", type=float, default=1.0,
                   help="Include previous expert as teacher only if its FDD domain "
                        "distance from the current batch < this threshold. "
                        "Recommended: 0.8-1.2")

    p.add_argument("--backbone_only_teacher", action="store_true", default=False,
                   help="Use ONLY the frozen backbone as teacher for PL. "
                        "Ignores all previous experts. Best baseline so far.")
    p.add_argument("--no_backbone_only_teacher", dest="backbone_only_teacher",
                   action="store_false")

    # ─── Oracle domain detection ───
    # Assigns corruptions to 4 known families instead of FDD.
    # Useful for evaluating cross-expert interactions under
    # perfect domain grouping.
    p.add_argument("--oracle_domains", action="store_true", default=False,
                   help="Use oracle domain grouping (4 families) instead of FDD. "
                        "noise=[gauss,shot,impulse], blur=[defocus,glass,motion,zoom], "
                        "weather=[snow,frost,fog,brightness], "
                        "digital=[contrast,elastic,pixelate,jpeg]")

    # backbone evaluation
    p.add_argument("--eval_backbone", action="store_true", default=False,
                   help="Evaluate frozen backbone on each domain before adaptation")

    # misc
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output")

    cfg = p.parse_args(args)

    # ── auto-set learning rate per dataset ──
    if cfg.lr is None:
        lr_map = {
            "imagenet_c": 1e-5,
            "cifar100_c": 1e-5,
            "imagenet_plus": 5e-4,
            "imagenet_plusplus": 5e-4,
            "acdc": 3e-4,
        }
        cfg.lr = lr_map[cfg.dataset]

    # ── auto-set num_classes ──
    if cfg.dataset == "cifar100_c":
        cfg.num_classes = 100
        cfg.img_size = 384          # paper resizes CIFAR to 384
        cfg.backbone = "vit_base_patch16_384"

    # ── entropy threshold κ × ln(C) ──
    cfg.entropy_threshold = cfg.confidence_threshold * math.log(cfg.num_classes)

    return cfg
