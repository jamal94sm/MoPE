"""
main.py — TENT Test-Time Adaptation.

Two modes:
  ImageNet-C:  Classification — report error rate per corruption
  CASIA-MS:    Verification — train ArcFace head, then TENT adapt

CASIA-MS flow:
  1. Load ArcFace backbone (ONNX)
  2. Evaluate frozen backbone on test spectrums (baseline)
  3. Train ArcFace head on train spectrums
  4. Evaluate trained model on test spectrums (post-training)
  5. TENT: adapt BN layers on test spectrums via entropy minimization
  6. Evaluate after TENT on test spectrums (post-TENT)
"""

import os, json, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_cfg, CASIA_ORACLE_LOOKUP, CASIA_ORACLE_DOMAINS
from backbones import build_model, ArcFaceHead
import tent
from datasets import (
    get_imagenet_c_loaders, get_casia_ms_train_test,
    split_gallery_probe, extract_embeddings, evaluate_verification,
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════
#  Shared evaluation helper
# ══════════════════════════════════════════════════════════════

def eval_all_test_spectrums(model, test_loaders, gallery_ratio, batch_size,
                             device, num_workers, seed, tag=""):
    """Evaluate EER + Rank-1 on each test spectrum. Returns dict."""
    was_training = model.training
    model.eval()
    results = {}
    for sname, loader, ds in test_loaders:
        gallery_idx, probe_idx = split_gallery_probe(ds, gallery_ratio, seed)
        all_idx = list(range(len(ds)))
        feats, labels = extract_embeddings(
            model, ds, all_idx, batch_size, device, num_workers)
        feats_t = feats.to(device)
        ver = evaluate_verification(feats_t, labels, gallery_idx, probe_idx)
        results[sname] = ver
        print(f"  {tag}{sname:>6s} → EER: {ver['eer']:.2f}% | "
              f"Rank-1: {ver['rank1']:.2f}% | "
              f"Gal: {ver['n_gallery']} | Probe: {ver['n_probe']}")
    mean_eer = np.mean([r['eer'] for r in results.values()])
    mean_r1 = np.mean([r['rank1'] for r in results.values()])
    print(f"  {tag}Mean EER: {mean_eer:.2f}% | Mean Rank-1: {mean_r1:.2f}%")
    if was_training:
        model.train()
    return results


# ══════════════════════════════════════════════════════════════
#  ImageNet-C (Classification)
# ══════════════════════════════════════════════════════════════

def adapt_imagenet_c(cfg):
    print(f"\n{'='*80}")
    print(f"  TENT — ImageNet-C Classification")
    print(f"  Backbone: {cfg.backbone} | LR: {cfg.tent_lr} | "
          f"Steps/batch: {cfg.tent_steps} | Episodic: {cfg.tent_episodic}")
    print(f"{'='*80}\n")

    model = build_model(cfg)
    norm_mean = getattr(cfg, '_norm_mean', (0.5, 0.5, 0.5))
    norm_std = getattr(cfg, '_norm_std', (0.5, 0.5, 0.5))

    loaders = get_imagenet_c_loaders(
        cfg.data_dir, cfg.severity, cfg.batch_size, cfg.num_workers,
        cfg.img_size, cfg.corruptions, list(norm_mean), list(norm_std))

    # Baseline
    baseline = {}
    if cfg.eval_backbone:
        print("[Baseline] Evaluating frozen backbone...")
        model.eval()
        with torch.no_grad():
            for cname, loader in loaders:
                correct = total = 0
                for imgs, labs in loader:
                    imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)
                    preds = model(imgs).argmax(1)
                    correct += (preds == labs).sum().item()
                    total += labs.shape[0]
                err = 100.0 * (1 - correct / total)
                baseline[cname] = err
                print(f"  {cname:25s} → {err:.1f}%")
        print(f"[Baseline] Mean: {np.mean(list(baseline.values())):.1f}%\n")

    # TENT
    model = tent.configure_model(model)
    tent.check_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
    tented_model = tent.Tent(model, optimizer,
                             steps=cfg.tent_steps,
                             episodic=cfg.tent_episodic)
    print(f"[TENT] {len(params)} BN params ({sum(p.numel() for p in params)} values)")

    results = {}
    for seg_idx, (cname, loader) in enumerate(loaders):
        if cfg.tent_episodic:
            tented_model.reset()

        n_batches = len(loader)
        seg_correct = seg_total = 0
        t0 = time.time()

        print(f"\n{'─'*70}")
        print(f"  [{seg_idx+1}/{len(loaders)}] {cname} ({len(loader.dataset)} samples)")
        print(f"{'─'*70}")
        print(f"  {'bat':>5} │{'err%':>6} │{'H':>6}")

        for batch_idx, (imgs, labs) in enumerate(loader):
            imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)
            logits = tented_model(imgs)
            preds = logits.argmax(1)
            correct = (preds == labs).sum().item()
            seg_correct += correct; seg_total += labs.shape[0]
            err = 100.0 * (1 - correct / labs.shape[0])

            if batch_idx < 5 or batch_idx % 100 == 0 or batch_idx == n_batches - 1:
                H = tent.softmax_entropy(logits).mean().item()
                print(f"  {batch_idx:5d} │{err:5.1f} │{H:6.3f}")

        seg_err = 100.0 * (1 - seg_correct / seg_total)
        elapsed = time.time() - t0
        results[cname] = seg_err

        b_err = baseline.get(cname)
        print(f"\n  ┌── SUMMARY: {cname}")
        if b_err is not None:
            imp = b_err - seg_err
            print(f"  │ Backbone: {b_err:.1f}% → TENT: {seg_err:.1f}% "
                  f"({'↓' if imp > 0 else '↑'}{abs(imp):.1f}%)")
        else:
            print(f"  │ TENT Error: {seg_err:.1f}%")
        print(f"  │ Time: {elapsed:.1f}s")
        print(f"  └{'─'*50}")

    _print_classification_summary(results, baseline, cfg)


def _print_classification_summary(results, baseline, cfg):
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Corruption':<25} ", end="")
    if baseline:
        print(f"{'Backbone':>10} {'TENT':>10} {'Δ':>10}")
    else:
        print(f"{'TENT':>10}")
    print(f"  {'─'*60}")

    for cname, terr in results.items():
        if baseline and cname in baseline:
            berr = baseline[cname]; imp = berr - terr
            print(f"  {cname:<25} {berr:>9.1f}% {terr:>9.1f}% "
                  f"{'↓' if imp > 0 else '↑'}{abs(imp):>8.1f}%")
        else:
            print(f"  {cname:<25} {terr:>9.1f}%")

    mean_tent = np.mean(list(results.values()))
    print(f"  {'─'*60}")
    if baseline:
        mean_base = np.mean(list(baseline.values()))
        imp = mean_base - mean_tent
        print(f"  {'MEAN':<25} {mean_base:>9.1f}% {mean_tent:>9.1f}% "
              f"{'↓' if imp > 0 else '↑'}{abs(imp):>8.1f}%")
    else:
        print(f"  {'MEAN':<25} {mean_tent:>9.1f}%")

    os.makedirs(cfg.output_dir, exist_ok=True)
    p = os.path.join(cfg.output_dir, f"imagenetc_{cfg.backbone}_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump({"tent": results, "mean_tent": mean_tent,
                    "backbone": cfg.backbone,
                    **({"baseline": baseline} if baseline else {})}, f, indent=2)
    print(f"\n  Saved: {p}")


# ══════════════════════════════════════════════════════════════
#  CASIA-MS (Verification)
# ══════════════════════════════════════════════════════════════

def adapt_casia_ms(cfg):
    print(f"\n{'='*80}")
    print(f"  TENT — CASIA-MS Palmprint Verification")
    print(f"  Backbone: ArcFace iResNet100")
    print(f"  ArcFace training: {cfg.arcface_epochs} epochs, "
          f"LR={cfg.arcface_lr} (head only, backbone frozen)")
    print(f"  TENT: LR={cfg.tent_lr}, steps={cfg.tent_steps}, "
          f"episodic={cfg.tent_episodic}")
    print(f"  Gallery ratio: {cfg.gallery_ratio}")
    if cfg.oracle_domains:
        print(f"  TENT domain grouping: ORACLE (3 groups)")
        for gn, specs in CASIA_ORACLE_DOMAINS.items():
            print(f"    {gn}: {specs}")
    print(f"{'='*80}\n")

    # ── Step 1: Build datasets ──
    identity_to_idx, num_identities, train_loader, test_loaders = \
        get_casia_ms_train_test(
            cfg.data_dir, cfg.train_spectrums, cfg.batch_size,
            cfg.num_workers, cfg.img_size)

    cfg.arcface_num_classes = num_identities

    # ── Step 2: Build model ──
    model = build_model(cfg)

    trainable = sum(p.numel() for p in model.head.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Head params: {trainable/1e6:.2f}M | "
          f"Total: {total/1e6:.2f}M (backbone frozen)")

    # ── Step 3: Evaluate frozen backbone (pre-training baseline) ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 1: Frozen Backbone Baseline")
    print(f"{'─'*70}")
    baseline = eval_all_test_spectrums(
        model, test_loaders, cfg.gallery_ratio, cfg.batch_size,
        cfg.device, cfg.num_workers, cfg.seed, tag="[baseline] ")

    # ── Step 4: Train ArcFace head ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 2: ArcFace Head Training ({cfg.arcface_epochs} epochs)")
    print(f"  Train spectrums: {cfg.train_spectrums}")
    print(f"  {num_identities} identity classes, "
          f"{len(train_loader.dataset)} train samples")
    print(f"{'─'*70}")

    # Only train the ArcFace head — backbone is frozen
    train_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=cfg.arcface_lr,
                                   weight_decay=cfg.arcface_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.arcface_epochs, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss()

    best_rank1 = 0.0
    ckpt_path = os.path.join(cfg.output_dir, "arcface_best.pth")
    os.makedirs(cfg.output_dir, exist_ok=True)

    model.train()
    for epoch in range(1, cfg.arcface_epochs + 1):
        ep_loss = 0.0; ep_corr = 0; ep_tot = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()

            logits = model.train_forward(imgs, labels)
            loss = ce_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)
            optimizer.step()

            ep_loss += loss.item()
            with torch.no_grad():
                ep_corr += (logits.argmax(1) == labels).sum().item()
                ep_tot += labels.shape[0]

        scheduler.step()
        acc = 100.0 * ep_corr / max(ep_tot, 1)
        n = len(train_loader)
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] ep {epoch:03d}/{cfg.arcface_epochs}  "
              f"loss={ep_loss/n:.4f}  acc={acc:.2f}%")

        if epoch % cfg.arcface_eval_every == 0 or epoch == cfg.arcface_epochs:
            print(f"  --- eval at epoch {epoch} ---")
            ver_results = eval_all_test_spectrums(
                model, test_loaders, cfg.gallery_ratio, cfg.batch_size,
                cfg.device, cfg.num_workers, cfg.seed, tag="  ")
            mean_r1 = np.mean([r['rank1'] for r in ver_results.values()])
            if mean_r1 > best_rank1:
                best_rank1 = mean_r1
                torch.save({
                    "epoch": epoch,
                    "model": model.backbone.state_dict(),
                    "arc": model.head.state_dict(),
                    "rank1": best_rank1,
                }, ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% → saved ***")

    # Load best checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model.backbone.load_state_dict(ckpt["model"])
        model.head.load_state_dict(ckpt["arc"])
        print(f"\n  Loaded best checkpoint (epoch {ckpt['epoch']}, "
              f"Rank-1={ckpt['rank1']:.2f}%)")

    # ── Step 5: Evaluate after training (pre-TENT) ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 3: Post-Training Evaluation")
    print(f"{'─'*70}")
    post_train = eval_all_test_spectrums(
        model, test_loaders, cfg.gallery_ratio, cfg.batch_size,
        cfg.device, cfg.num_workers, cfg.seed, tag="[trained] ")

    # ── Step 6: TENT adaptation on test spectrums ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 4: TENT Adaptation on Test Spectrums")
    print(f"{'─'*70}")

    model = tent.configure_model(model)
    tent.check_model(model)
    params, param_names = tent.collect_params(model)
    tent_optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
    tented_model = tent.Tent(model, tent_optimizer,
                              steps=cfg.tent_steps,
                              episodic=cfg.tent_episodic)

    print(f"[TENT] {len(params)} BN params "
          f"({sum(p.numel() for p in params)} values)")

    # Group spectrums by oracle domain if enabled
    if cfg.oracle_domains:
        from collections import OrderedDict
        groups = OrderedDict()
        for sname, loader, ds in test_loaders:
            gname, gid = CASIA_ORACLE_LOOKUP.get(sname, ("unk", -1))
            if gid not in groups:
                groups[gid] = {"name": gname, "spectrums": []}
            groups[gid]["spectrums"].append((sname, loader, ds))
        domain_sequence = [(gi["name"], gid, gi["spectrums"])
                           for gid, gi in groups.items()]
    else:
        domain_sequence = [(s, i, [(s, ld, ds)])
                           for i, (s, ld, ds) in enumerate(test_loaders)]

    for dom_idx, (dom_name, dom_id, spectrum_list) in enumerate(domain_sequence):
        if cfg.tent_episodic:
            tented_model.reset()

        total_batches = sum(len(ld) for _, ld, _ in spectrum_list)
        spec_names = [s for s, _, _ in spectrum_list]
        t0 = time.time()

        print(f"\n  [{dom_idx+1}/{len(domain_sequence)}] Domain: {dom_name} "
              f"({spec_names})")
        print(f"  {'bat':>5} │{'spec':>6} │{'H':>6}")

        global_batch = 0
        for sname, loader, ds in spectrum_list:
            for batch_idx, (imgs, labs) in enumerate(loader):
                imgs = imgs.to(cfg.device)
                logits = tented_model(imgs)

                if global_batch < 5 or global_batch % 50 == 0 or \
                   global_batch == total_batches - 1:
                    H = tent.softmax_entropy(logits).mean().item()
                    print(f"  {global_batch:5d} │{sname:>6s} │{H:6.3f}")
                global_batch += 1

        print(f"  Time: {time.time() - t0:.1f}s")

    # ── Step 7: Evaluate after TENT ──
    print(f"\n{'─'*70}")
    print(f"  PHASE 5: Post-TENT Evaluation")
    print(f"{'─'*70}")
    model.eval()
    post_tent = eval_all_test_spectrums(
        model, test_loaders, cfg.gallery_ratio, cfg.batch_size,
        cfg.device, cfg.num_workers, cfg.seed, tag="[tent] ")

    # ── Final comparison table ──
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON: CASIA-MS Verification")
    print(f"  Train spectrums: {cfg.train_spectrums}")
    print(f"{'='*80}")
    print(f"\n  {'Spectrum':<10} {'BB EER':>8} {'BB R1':>8} "
          f"{'Trn EER':>8} {'Trn R1':>8} "
          f"{'TNT EER':>8} {'TNT R1':>8}")
    print(f"  {'─'*65}")

    for sname in post_tent:
        b = baseline.get(sname, {})
        t = post_train.get(sname, {})
        n = post_tent[sname]
        print(f"  {sname:<10} "
              f"{b.get('eer', -1):>7.2f}% {b.get('rank1', -1):>7.2f}% "
              f"{t.get('eer', -1):>7.2f}% {t.get('rank1', -1):>7.2f}% "
              f"{n['eer']:>7.2f}% {n['rank1']:>7.2f}%")

    def _mean(d, k):
        return np.mean([r[k] for r in d.values()]) if d else -1

    print(f"  {'─'*65}")
    print(f"  {'MEAN':<10} "
          f"{_mean(baseline,'eer'):>7.2f}% {_mean(baseline,'rank1'):>7.2f}% "
          f"{_mean(post_train,'eer'):>7.2f}% {_mean(post_train,'rank1'):>7.2f}% "
          f"{_mean(post_tent,'eer'):>7.2f}% {_mean(post_tent,'rank1'):>7.2f}%")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_data = {
        "baseline": {k: dict(v) for k, v in baseline.items()},
        "post_train": {k: dict(v) for k, v in post_train.items()},
        "post_tent": {k: dict(v) for k, v in post_tent.items()},
        "train_spectrums": cfg.train_spectrums,
        "arcface_epochs": cfg.arcface_epochs,
        "tent_lr": cfg.tent_lr,
        "gallery_ratio": cfg.gallery_ratio,
    }
    p = os.path.join(cfg.output_dir, f"casia_ms_full_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")
    if os.path.exists(ckpt_path):
        print(f"  Best ArcFace checkpoint: {ckpt_path}")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = get_cfg()
    set_seed(cfg.seed)

    if cfg.dataset == "casia_ms":
        adapt_casia_ms(cfg)
    elif cfg.dataset == "imagenet_c":
        adapt_imagenet_c(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
