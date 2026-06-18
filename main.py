"""
main.py — TENT Test-Time Adaptation.

Two modes:
  ImageNet-C:  Classification — report error rate per corruption
  CASIA-MS:    Verification   — report EER + Rank-1 per spectrum

Flow:
  1. Load pretrained model
  2. (Optional) Evaluate frozen backbone baseline
  3. Configure model for TENT (BN train, everything else frozen)
  4. For each domain: adapt BN params via entropy minimization
  5. Evaluate after adaptation
"""

import os, json, time, random
import numpy as np
import torch
import torch.nn.functional as F

from config import get_cfg, CASIA_ORACLE_LOOKUP, CASIA_ORACLE_DOMAINS
from backbones import build_model
import tent
from datasets import (
    get_imagenet_c_loaders, get_casia_ms_loaders,
    split_gallery_probe, extract_embeddings, evaluate_verification,
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════
#  ImageNet-C (Classification)
# ══════════════════════════════════════════════════════════════

def adapt_imagenet_c(cfg):
    print(f"\n{'='*80}")
    print(f"  TENT — ImageNet-C Classification")
    print(f"  Backbone: {cfg.backbone} | LR: {cfg.tent_lr} | "
          f"Steps/batch: {cfg.tent_steps} | "
          f"Episodic: {cfg.tent_episodic}")
    print(f"{'='*80}\n")

    model = build_model(cfg)

    norm_mean = getattr(cfg, '_norm_mean', (0.5, 0.5, 0.5))
    norm_std = getattr(cfg, '_norm_std', (0.5, 0.5, 0.5))

    loaders = get_imagenet_c_loaders(
        cfg.data_dir, cfg.severity, cfg.batch_size, cfg.num_workers,
        cfg.img_size, cfg.corruptions, list(norm_mean), list(norm_std))

    # ── Baseline evaluation ──
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

    # ── Configure TENT ──
    model = tent.configure_model(model)
    tent.check_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
    tented_model = tent.Tent(model, optimizer,
                             steps=cfg.tent_steps,
                             episodic=cfg.tent_episodic)

    print(f"[TENT] {len(params)} BN parameters "
          f"({sum(p.numel() for p in params)} values)")

    results = {}

    for seg_idx, (cname, loader) in enumerate(loaders):
        if cfg.tent_episodic:
            tented_model.reset()

        n_batches = len(loader)
        seg_correct = seg_total = 0
        t0 = time.time()

        print(f"\n{'─'*70}")
        print(f"  [{seg_idx+1}/{len(loaders)}] {cname} "
              f"({len(loader.dataset)} samples)")
        print(f"{'─'*70}")
        print(f"  {'bat':>5} │{'err%':>6} │{'H':>6}")

        for batch_idx, (imgs, labs) in enumerate(loader):
            imgs, labs = imgs.to(cfg.device), labs.to(cfg.device)

            logits = tented_model(imgs)

            preds = logits.argmax(1)
            correct = (preds == labs).sum().item()
            seg_correct += correct
            seg_total += labs.shape[0]
            err = 100.0 * (1 - correct / labs.shape[0])

            if batch_idx < 5 or batch_idx % 100 == 0 or \
               batch_idx == n_batches - 1:
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

    # ── Final summary ──
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
    save_data = {"tent": results, "mean_tent": mean_tent,
                 "backbone": cfg.backbone, "tent_lr": cfg.tent_lr}
    if baseline:
        save_data["baseline"] = baseline
        save_data["mean_baseline"] = mean_base
    p = os.path.join(cfg.output_dir, f"imagenetc_{cfg.backbone}_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")


# ══════════════════════════════════════════════════════════════
#  CASIA-MS (Verification)
# ══════════════════════════════════════════════════════════════

def adapt_casia_ms(cfg):
    print(f"\n{'='*80}")
    print(f"  TENT — CASIA-MS Palmprint Verification")
    print(f"  Backbone: ArcFace iResNet100 | LR: {cfg.tent_lr} | "
          f"Steps/batch: {cfg.tent_steps}")
    print(f"  Gallery ratio: {cfg.gallery_ratio} | "
          f"Episodic: {cfg.tent_episodic}")
    if cfg.oracle_domains:
        print(f"  Domain grouping: ORACLE (3 groups)")
        for gn, specs in CASIA_ORACLE_DOMAINS.items():
            print(f"    {gn}: {specs}")
    else:
        print(f"  Domain grouping: per-spectrum")
    print(f"{'='*80}\n")

    model = build_model(cfg)

    loaders = get_casia_ms_loaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers,
        cfg.img_size, cfg.casia_spectrums)

    print(f"[Data] {len(loaders)} spectrums:")
    for sname, loader, ds in loaders:
        print(f"  {sname}: {len(ds)} samples, {ds.num_identities} IDs")

    # ── Baseline evaluation ──
    baseline = {}
    if cfg.eval_backbone:
        print(f"\n[Baseline] Evaluating frozen backbone on each spectrum...")
        model.eval()
        for sname, loader, ds in loaders:
            gallery_idx, probe_idx = split_gallery_probe(
                ds, cfg.gallery_ratio, cfg.seed)
            all_idx = list(range(len(ds)))
            feats, labels = extract_embeddings(
                model, ds, all_idx, cfg.batch_size, cfg.device,
                cfg.num_workers)
            feats_t = feats.to(cfg.device)
            result = evaluate_verification(
                feats_t, labels, gallery_idx, probe_idx)
            baseline[sname] = result
            print(f"  {sname:>6s} → EER: {result['eer']:.2f}% | "
                  f"Rank-1: {result['rank1']:.2f}% | "
                  f"Gal: {result['n_gallery']} | Probe: {result['n_probe']}")
        mean_eer = np.mean([r['eer'] for r in baseline.values()])
        mean_r1 = np.mean([r['rank1'] for r in baseline.values()])
        print(f"[Baseline] Mean EER: {mean_eer:.2f}% | "
              f"Mean Rank-1: {mean_r1:.2f}%\n")

    # ── Configure TENT ──
    model = tent.configure_model(model)
    tent.check_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
    tented_model = tent.Tent(model, optimizer,
                             steps=cfg.tent_steps,
                             episodic=cfg.tent_episodic)

    print(f"[TENT] {len(params)} BN parameters "
          f"({sum(p.numel() for p in params)} values)")

    # ── Group spectrums by oracle domain if enabled ──
    if cfg.oracle_domains:
        from collections import OrderedDict
        groups = OrderedDict()
        for sname, loader, ds in loaders:
            gname, gid = CASIA_ORACLE_LOOKUP.get(sname, ("unk", -1))
            if gid not in groups:
                groups[gid] = {"name": gname, "spectrums": []}
            groups[gid]["spectrums"].append((sname, loader, ds))
        domain_sequence = []
        for gid, ginfo in groups.items():
            domain_sequence.append((ginfo["name"], gid, ginfo["spectrums"]))
    else:
        domain_sequence = [(sname, i, [(sname, loader, ds)])
                           for i, (sname, loader, ds) in enumerate(loaders)]

    results = {}

    for dom_idx, (dom_name, dom_id, spectrum_list) in enumerate(domain_sequence):
        if cfg.tent_episodic:
            tented_model.reset()

        total_samples = sum(len(ds) for _, _, ds in spectrum_list)
        total_batches = sum(len(ld) for _, ld, _ in spectrum_list)
        spec_names = [s for s, _, _ in spectrum_list]

        print(f"\n{'─'*70}")
        print(f"  [{dom_idx+1}/{len(domain_sequence)}] Domain: {dom_name} "
              f"(spectrums: {spec_names}, {total_samples} samples)")
        print(f"{'─'*70}")
        print(f"  {'bat':>5} │{'spec':>6} │{'H':>6}")

        t0 = time.time()
        global_batch = 0

        # ── Adapt on all spectrums in this domain ──
        for sname, loader, ds in spectrum_list:
            for batch_idx, (imgs, labs) in enumerate(loader):
                imgs = imgs.to(cfg.device)

                logits = tented_model(imgs)

                if global_batch < 5 or global_batch % 50 == 0 or \
                   global_batch == total_batches - 1:
                    H = tent.softmax_entropy(logits).mean().item()
                    print(f"  {global_batch:5d} │{sname:>6s} │{H:6.3f}")
                global_batch += 1

        elapsed = time.time() - t0

        # ── Evaluate after adaptation ──
        print(f"\n  [Evaluating after TENT on {dom_name}...]")
        # Switch to eval for embedding extraction
        # (BN will use batch stats since track_running_stats=False)
        model.eval()

        for sname, loader, ds in spectrum_list:
            gallery_idx, probe_idx = split_gallery_probe(
                ds, cfg.gallery_ratio, cfg.seed)
            all_idx = list(range(len(ds)))
            feats, labels = extract_embeddings(
                model, ds, all_idx, cfg.batch_size, cfg.device,
                cfg.num_workers)
            feats_t = feats.to(cfg.device)
            ver = evaluate_verification(feats_t, labels, gallery_idx, probe_idx)
            results[sname] = ver

            b = baseline.get(sname)
            print(f"\n  ┌── {sname}")
            if b:
                de = b["eer"] - ver["eer"]
                dr = ver["rank1"] - b["rank1"]
                print(f"  │ Backbone → EER: {b['eer']:.2f}% | "
                      f"Rank-1: {b['rank1']:.2f}%")
                print(f"  │ TENT    → EER: {ver['eer']:.2f}% | "
                      f"Rank-1: {ver['rank1']:.2f}%")
                print(f"  │ Change: EER {'↓' if de > 0 else '↑'}"
                      f"{abs(de):.2f}% | Rank-1 {'↑' if dr > 0 else '↓'}"
                      f"{abs(dr):.2f}%")
            else:
                print(f"  │ EER: {ver['eer']:.2f}% | "
                      f"Rank-1: {ver['rank1']:.2f}%")
            print(f"  │ Gallery: {ver['n_gallery']} | "
                  f"Probe: {ver['n_probe']}")
            print(f"  └{'─'*50}")

        # Back to train mode for next domain
        model.train()
        print(f"  Time: {elapsed:.1f}s")

    # ── Final summary ──
    _print_verification_summary(results, baseline, cfg)


def _print_verification_summary(results, baseline, cfg):
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS: CASIA-MS Verification")
    print(f"{'='*80}")
    print(f"\n  {'Spectrum':<10} ", end="")
    if baseline:
        print(f"{'BB EER':>8} {'BB R1':>8} {'TENT EER':>9} {'TENT R1':>8} "
              f"{'ΔEER':>8} {'ΔR1':>8}")
    else:
        print(f"{'EER':>8} {'Rank-1':>8}")
    print(f"  {'─'*75}")

    for sname, r in results.items():
        if baseline and sname in baseline:
            b = baseline[sname]
            de = b["eer"] - r["eer"]
            dr = r["rank1"] - b["rank1"]
            print(f"  {sname:<10} {b['eer']:>7.2f}% {b['rank1']:>7.2f}% "
                  f"{r['eer']:>8.2f}% {r['rank1']:>7.2f}% "
                  f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
                  f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")
        else:
            print(f"  {sname:<10} {r['eer']:>7.2f}% {r['rank1']:>7.2f}%")

    mean_eer = np.mean([r['eer'] for r in results.values()])
    mean_r1 = np.mean([r['rank1'] for r in results.values()])
    print(f"  {'─'*75}")
    if baseline:
        bm_eer = np.mean([r['eer'] for r in baseline.values()])
        bm_r1 = np.mean([r['rank1'] for r in baseline.values()])
        de = bm_eer - mean_eer
        dr = mean_r1 - bm_r1
        print(f"  {'MEAN':<10} {bm_eer:>7.2f}% {bm_r1:>7.2f}% "
              f"{mean_eer:>8.2f}% {mean_r1:>7.2f}% "
              f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
              f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")
    else:
        print(f"  {'MEAN':<10} {mean_eer:>7.2f}% {mean_r1:>7.2f}%")

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_data = {"tent": {k: dict(v) for k, v in results.items()},
                 "mean_eer": mean_eer, "mean_rank1": mean_r1,
                 "tent_lr": cfg.tent_lr, "gallery_ratio": cfg.gallery_ratio}
    if baseline:
        save_data["baseline"] = {k: dict(v) for k, v in baseline.items()}
    p = os.path.join(cfg.output_dir, f"casia_ms_tent_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")


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
