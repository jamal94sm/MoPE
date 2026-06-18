"""
pseudo_labels.py — Cross-expert pseudo-label supervision.

v7: No KD. No shared expert voter. Hard or soft PL with adjustable sharpness.
    Voters: backbone + previous domain experts (FDD distance gated).
    Dists column shows distance to each known FDD domain.
"""

import torch
import torch.nn.functional as F
from collections import Counter


@torch.no_grad()
def get_teacher_signals(model, images, active_domain, cfg,
                        fdd_distances=None):
    """
    Voters: backbone + previous domain experts (FDD distance gated).
    No shared expert, no KD.

    Returns: pseudo_labels, pl_mask, teacher_probs, stats
    """
    B = images.shape[0]
    device = images.device
    threshold = cfg.pl_threshold
    agreement_ratio = cfg.pl_agreement
    include_thresh = cfg.fdd_include_threshold

    # ─── Backbone-only mode (best baseline) ───────────────────────────
    if cfg.backbone_only_teacher:
        bb_logits = model.backbone(images)
        bb_probs = F.softmax(bb_logits, dim=-1)
        bb_conf, bb_pred = bb_probs.max(dim=-1)

        pl_mask = bb_conf > threshold
        pseudo_labels = bb_pred

        T = cfg.pl_sharpness if cfg.pl_soft else 1.0
        teacher_probs = F.softmax(bb_logits / T, dim=-1)

        if fdd_distances:
            dist_str = "/".join(f"d{k}={v:.1f}" for k, v in sorted(fdd_distances.items()))
        else:
            dist_str = "--"

        stats = {
            "num_voters": 1,
            "num_agreed": pl_mask.sum().item(),
            "agreement_rate": pl_mask.float().mean().item() * 100,
            "experts_excluded": 0, "experts_included": 0,
            "bb_pred": bb_pred, "teacher_pred": bb_pred,
            "teach_str": "bb", "dist_str": dist_str,
            "excluded_details": [],
        }
        return pseudo_labels, pl_mask, teacher_probs, stats

    # ─── Full mode: backbone + FDD-gated previous experts ─────────────

    voter_names = []
    all_preds = []
    all_confs = []
    all_logits = []

    # ─── Voter 0: frozen backbone (always included) ───────────────────
    bb_logits = model.backbone(images)
    bb_probs = F.softmax(bb_logits, dim=-1)
    bb_conf, bb_pred = bb_probs.max(dim=-1)

    all_logits.append(bb_logits)
    all_preds.append(bb_pred)
    all_confs.append(bb_conf)
    voter_names.append("bb")

    # ─── Voters 1+: previous domain experts (FDD distance gated) ──────
    saved_domain = active_domain
    included_experts = []
    excluded_experts = []

    for dom_id in range(active_domain):
        if fdd_distances is not None and dom_id in fdd_distances:
            dist = fdd_distances[dom_id]
            if dist > include_thresh:
                excluded_experts.append((dom_id, dist))
                continue
            included_experts.append((dom_id, dist))
        else:
            included_experts.append((dom_id, -1.0))

        model.set_active_domain(dom_id)
        exp_logits = model(images)
        exp_probs = F.softmax(exp_logits, dim=-1)
        exp_conf, exp_pred = exp_probs.max(dim=-1)

        all_logits.append(exp_logits)
        all_preds.append(exp_pred)
        all_confs.append(exp_conf)
        voter_names.append(f"e{dom_id}")

    model.set_active_domain(saved_domain)

    num_voters = len(all_preds)
    preds_stack = torch.stack(all_preds, dim=0)
    confs_stack = torch.stack(all_confs, dim=0)
    logits_stack = torch.stack(all_logits, dim=0)

    # ─── Group consensus ──────────────────────────────────────────────
    pseudo_labels = torch.zeros(B, dtype=torch.long, device=device)
    pl_mask = torch.zeros(B, dtype=torch.bool, device=device)
    teacher_probs = torch.zeros(B, bb_probs.shape[-1], device=device)
    min_agree = max(1, int(num_voters * agreement_ratio))

    # sharpness for soft PL teacher distribution
    T = cfg.pl_sharpness if cfg.pl_soft else 1.0
    soft_stack = F.softmax(logits_stack / T, dim=-1)

    for i in range(B):
        sample_preds = preds_stack[:, i]
        sample_confs = confs_stack[:, i]
        sample_soft = soft_stack[:, i, :]

        confident = sample_confs > threshold
        n_confident = confident.sum().item()

        if n_confident == 0:
            teacher_probs[i] = soft_stack[0, i, :]
            continue

        confident_preds = sample_preds[confident].cpu().tolist()
        counts = Counter(confident_preds)
        majority_class, majority_count = counts.most_common(1)[0]

        agrees = (sample_preds == majority_class)
        in_group = confident & agrees
        n_in_group = in_group.sum().item()

        if n_in_group >= min_agree:
            pseudo_labels[i] = majority_class
            pl_mask[i] = True

        if n_in_group > 0:
            w = sample_confs[in_group]
            w = w / w.sum()
            teacher_probs[i] = (w[:, None] * sample_soft[in_group]).sum(dim=0)
        else:
            teacher_probs[i] = soft_stack[0, i, :]

    teacher_pred = teacher_probs.argmax(dim=-1)

    # teacher composition: "bb" or "bb+e0" or "bb+e0+e2"
    teach_str = "+".join(voter_names)

    # distances string: "d0=1.50/d1=0.82" for all known domains
    if fdd_distances:
        dist_str = "/".join(f"d{k}={v:.1f}" for k, v in sorted(fdd_distances.items()))
    else:
        dist_str = "--"

    stats = {
        "num_voters": num_voters,
        "num_agreed": pl_mask.sum().item(),
        "agreement_rate": pl_mask.float().mean().item() * 100,
        "experts_excluded": len(excluded_experts),
        "experts_included": len(included_experts),
        "bb_pred": bb_pred,
        "teacher_pred": teacher_pred,
        "teach_str": teach_str,
        "dist_str": dist_str,
        "excluded_details": excluded_experts,
    }

    return pseudo_labels, pl_mask, teacher_probs, stats


def compute_pl_loss(logits, pseudo_labels, pl_mask, teacher_probs, cfg):
    """
    Hard PL: CE with argmax class (one-hot target)
    Soft PL: CE with teacher's sharpened distribution
    """
    if cfg.pl_lambda <= 0 or pl_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), {
            "pl_loss": 0.0, "pl_samples": 0}

    if cfg.pl_soft:
        # Soft PL: -Σ teacher_c × log(student_c)
        student_log_probs = F.log_softmax(logits[pl_mask], dim=-1)
        pl_loss = -(teacher_probs[pl_mask] * student_log_probs).sum(dim=-1).mean()
    else:
        # Hard PL: standard cross-entropy
        pl_loss = F.cross_entropy(logits[pl_mask], pseudo_labels[pl_mask])

    return pl_loss, {
        "pl_loss": pl_loss.item(),
        "pl_samples": pl_mask.sum().item()}
