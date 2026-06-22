"""
main.py — Test-Time Adaptation: Contrastive + JEPA.

CASIA-MS pipeline:
  Phase 1: Train backbone (ArcFace) on source × train_IDs
  Phase 3: Baseline eval on target × test_IDs
  Phase 4+5: TTA (contrastive or JEPA) → eval per domain
"""

import os, json, time, random
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
#  Helpers
# ══════════════════════════════════════════════════════════════

def eval_test_spectrums(model, test_loaders, cfg, tag=""):
    was_training = model.training
    model.eval()
    results = {}
    for sname, loader, ds in test_loaders:
        gallery_idx, probe_idx = split_gallery_probe(
            ds, cfg.gallery_ratio, cfg.seed)
        all_idx = list(range(len(ds)))
        feats, labels = extract_embeddings(
            model, ds, all_idx, cfg.batch_size, cfg.device, cfg.num_workers)
        feats_t = feats.to(cfg.device)
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


def train_arcface(model, train_loader, train_params, cfg, epochs, tag,
                  test_loaders=None, ckpt_path=None, lr=None):
    lr = lr or cfg.arcface_lr
    optimizer = torch.optim.AdamW(train_params, lr=lr,
                                   weight_decay=cfg.arcface_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss()
    best_rank1 = 0.0
    model.train()

    for epoch in range(1, epochs + 1):
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
        print(f"  [{ts}] {tag} ep {epoch:03d}/{epochs}  "
              f"loss={ep_loss/n:.4f}  acc={acc:.2f}%")

        if test_loaders and (epoch % cfg.arcface_eval_every == 0 or
                             epoch == epochs):
            print(f"  --- eval at epoch {epoch} ---")
            ver = eval_test_spectrums(model, test_loaders, cfg, tag="  ")
            model.train()
            mean_r1 = np.mean([r['rank1'] for r in ver.values()])
            if mean_r1 > best_rank1 and ckpt_path:
                best_rank1 = mean_r1
                torch.save({
                    "epoch": epoch,
                    "backbone": model.backbone.state_dict(),
                    "head": model.head.state_dict(),
                    "rank1": best_rank1,
                }, ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% → saved ***")
    return best_rank1


# ══════════════════════════════════════════════════════════════
#  CASIA-MS Adaptation
# ══════════════════════════════════════════════════════════════

def adapt_casia_ms(cfg):
    method_label = cfg.tta_method.upper()
    print(f"\n{'='*80}")
    print(f"  TTA — CASIA-MS Verification | Method: {method_label}")
    print(f"  ID split: {100*(1-cfg.test_id_ratio):.0f}% train / "
          f"{100*cfg.test_id_ratio:.0f}% test")
    print(f"  Phase 1: Train backbone ({cfg.arcface_epochs} ep)")
    print(f"  Phase 4: {method_label} on target domains")
    print(f"  Mode: {'episodic' if cfg.reset_tta else 'continual'}")
    print(f"{'='*80}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Build datasets ──
    (train_ids, test_ids, train_id_map, test_id_map,
     backbone_train_loader, test_head_train_loader,
     test_loaders) = get_casia_ms_train_test(
        cfg.data_dir, cfg.train_spectrums, cfg.batch_size,
        cfg.num_workers, cfg.img_size, cfg.test_id_ratio, cfg.seed)

    n_train_cls = len(train_id_map)
    n_test_cls = len(test_id_map)

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: Train backbone (+ JEPA if jepa_joint mode)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    if cfg.tta_method == "jepa_joint":
        print(f"  PHASE 1: Train backbone + head_A + JEPA predictor (joint)")
        print(f"  Loss = ArcFace + {cfg.jepa_train_lambda} × JEPA")
    else:
        print(f"  PHASE 1: Train backbone + head_A ({n_train_cls} classes)")
    print(f"{'─'*70}")

    cfg.arcface_num_classes = n_train_cls
    model = build_model(cfg)

    train_params = ([p for p in model.backbone.parameters() if p.requires_grad]
                    + list(model.head.parameters()))
    n_unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"[Phase 1] {n_unfrozen} unfrozen backbone tensors + "
          f"head_A ({n_train_cls}×512)")

    # JEPA components (created during Phase 1, reused in Phase 4)
    warm_predictor_state = None

    if cfg.tta_method == "jepa_joint":
        from copy import deepcopy
        emb_dim = 512
        jepa_predictor = tent.PredictorMLP(
            emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
        jepa_teacher = deepcopy(model)
        jepa_teacher.requires_grad_(False)
        jepa_teacher.eval()

        jepa_loss_fn = (F.smooth_l1_loss if cfg.jepa_loss == "smooth_l1"
                        else F.mse_loss)
        jepa_aug = tent.get_tta_augmentation(cfg.img_size)

        train_params += list(jepa_predictor.parameters())
        print(f"[Phase 1] + JEPA predictor: "
              f"{sum(p.numel() for p in jepa_predictor.parameters())} params | "
              f"EMA: {cfg.jepa_momentum}")

    # Training loop
    optimizer = torch.optim.AdamW(train_params, lr=cfg.arcface_lr,
                                   weight_decay=cfg.arcface_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.arcface_epochs, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss()
    ckpt1 = os.path.join(cfg.output_dir, "phase1_best.pth")
    best_rank1 = 0.0

    model.train()
    for epoch in range(1, cfg.arcface_epochs + 1):
        ep_loss = 0.0; ep_arc = 0.0; ep_jepa = 0.0
        ep_corr = 0; ep_tot = 0; n_bat = 0

        for imgs, labels in backbone_train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()

            # ArcFace loss
            logits = model.train_forward(imgs, labels)
            loss_arc = ce_loss(logits, labels)
            total_loss = loss_arc

            # JEPA loss (joint training)
            if cfg.tta_method == "jepa_joint":
                z_s = model.get_raw_embeddings(imgs)
                with torch.no_grad():
                    x_aug = tent.augment_batch(
                        imgs, jepa_aug,
                        use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta)
                    z_t = jepa_teacher.get_raw_embeddings(x_aug)
                z_p = jepa_predictor(z_s)
                loss_jepa = jepa_loss_fn(z_p, z_t)
                total_loss = total_loss + cfg.jepa_train_lambda * loss_jepa
                ep_jepa += loss_jepa.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 5.0)
            optimizer.step()

            # EMA teacher update
            if cfg.tta_method == "jepa_joint":
                with torch.no_grad():
                    tent.ema_update(model, jepa_teacher, cfg.jepa_momentum)

            ep_arc += loss_arc.item()
            with torch.no_grad():
                ep_corr += (logits.argmax(1) == labels).sum().item()
                ep_tot += labels.shape[0]
            n_bat += 1

        scheduler.step()
        acc = 100.0 * ep_corr / max(ep_tot, 1)
        ts = time.strftime("%H:%M:%S")
        if cfg.tta_method == "jepa_joint":
            print(f"  [{ts}] P1 ep {epoch:03d}/{cfg.arcface_epochs}  "
                  f"arc={ep_arc/n_bat:.4f}  jepa={ep_jepa/n_bat:.4f}  "
                  f"acc={acc:.2f}%")
        else:
            print(f"  [{ts}] P1 ep {epoch:03d}/{cfg.arcface_epochs}  "
                  f"loss={ep_arc/n_bat:.4f}  acc={acc:.2f}%")

    # Save warm predictor state
    if cfg.tta_method == "jepa_joint":
        warm_predictor_state = deepcopy(jepa_predictor.state_dict())
        with torch.no_grad():
            sim = F.cosine_similarity(z_p, z_t, dim=-1).mean().item()
        print(f"\n  JEPA predictor trained: sim={sim:.3f}")

    model.backbone.requires_grad_(False)

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: JEPA warm-up (jepa / jepa_con modes)
    #  Frozen backbone, train only predictor + EMA teacher
    # ══════════════════════════════════════════════════════════
    if cfg.tta_method in ("jepa", "jepa_con") and cfg.jepa_warmup_epochs > 0:
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: JEPA predictor warm-up on source")
        print(f"  {cfg.jepa_warmup_epochs} epochs, backbone FROZEN")
        print(f"{'─'*70}")

        from copy import deepcopy
        emb_dim = 512

        warm_predictor = tent.PredictorMLP(
            emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
        warm_teacher = deepcopy(model)
        warm_teacher.requires_grad_(False)
        warm_teacher.eval()

        jepa_loss_fn = (F.smooth_l1_loss if cfg.jepa_loss == "smooth_l1"
                        else F.mse_loss)
        aug_tf = tent.get_tta_augmentation(cfg.img_size)

        warmup_opt = torch.optim.Adam(warm_predictor.parameters(),
                                       lr=cfg.tent_lr * 10)

        def _get_raw(mdl, x):
            if hasattr(mdl, 'get_raw_embeddings'):
                return mdl.get_raw_embeddings(x)
            elif hasattr(mdl, 'backbone') and \
                    hasattr(mdl.backbone, 'forward_raw'):
                return mdl.backbone.forward_raw(x)
            return mdl.get_embeddings(x)

        model.eval()
        for ep in range(1, cfg.jepa_warmup_epochs + 1):
            ep_loss = 0.0; n_bat = 0
            for imgs, _ in backbone_train_loader:
                imgs = imgs.to(cfg.device)
                with torch.no_grad():
                    z_s = _get_raw(model, imgs)
                    x_aug = tent.augment_batch(
                        imgs, aug_tf,
                        use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta)
                    z_t = _get_raw(warm_teacher, x_aug)

                z_p = warm_predictor(z_s)
                loss = jepa_loss_fn(z_p, z_t)

                warmup_opt.zero_grad()
                loss.backward()
                warmup_opt.step()

                with torch.no_grad():
                    tent.ema_update(model, warm_teacher, cfg.jepa_momentum)

                ep_loss += loss.item()
                n_bat += 1

            with torch.no_grad():
                sim = F.cosine_similarity(z_p, z_t, dim=-1).mean().item()
                p_std = z_p.std(dim=0).mean().item()
            print(f"  [Warm-up] ep {ep:02d}/{cfg.jepa_warmup_epochs}  "
                  f"loss={ep_loss/n_bat:.4f}  sim={sim:.3f}  "
                  f"p_std={p_std:.3f}")

        warm_predictor_state = deepcopy(warm_predictor.state_dict())
        print(f"  Predictor ready: sim={sim:.3f}")

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: Baseline eval
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 3: Pre-TTA Baseline")
    print(f"{'─'*70}")
    baseline = eval_test_spectrums(model, test_loaders, cfg,
                                    tag="[pre-TTA] ")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4+5: TTA — adapt → evaluate (per domain)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 4+5: {method_label} "
          f"({'episodic' if cfg.reset_tta else 'continual'})")
    print(f"{'─'*70}")

    # Domain sequence
    if cfg.oracle_domains:
        from collections import OrderedDict
        groups = OrderedDict()
        for sn, ld, ds in test_loaders:
            gn, gid = CASIA_ORACLE_LOOKUP.get(sn, ("unk", -1))
            if gid not in groups:
                groups[gid] = {"name": gn, "spectrums": []}
            groups[gid]["spectrums"].append((sn, ld, ds))
        dom_seq = [(g["name"], gi, g["spectrums"])
                   for gi, g in groups.items()]
    else:
        dom_seq = [(s, i, [(s, ld, ds)])
                   for i, (s, ld, ds) in enumerate(test_loaders)]

    from copy import deepcopy
    pre_adapt_state = deepcopy(model.state_dict())

    post_tta = {}

    # One-time setup for continual mode
    if not cfg.reset_tta:
        model = (tent.configure_model_safe(model) if cfg.safe_bn
                 else tent.configure_model(model))
        bn_params, _ = tent.collect_params(model)
        aug_tf = tent.get_tta_augmentation(cfg.img_size)

        if cfg.tta_method == "contrastive":
            opt = torch.optim.Adam(bn_params, lr=cfg.tent_lr)
            tta_obj = tent.Contrastive(
                model, opt, aug_tf,
                contrastive_lambda=cfg.contrastive_lambda,
                contrastive_temp=cfg.contrastive_temp,
                use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                steps=cfg.tent_steps)
            print(f"[CONTRASTIVE] {len(bn_params)} BN params")

        elif cfg.tta_method in ("jepa", "jepa_joint"):
            emb_dim = 512
            predictor = tent.PredictorMLP(
                emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
            if warm_predictor_state is not None:
                predictor.load_state_dict(warm_predictor_state)
                print(f"[{method_label}] Loaded warm predictor")
            opt = torch.optim.Adam([
                {"params": bn_params, "lr": cfg.tent_lr},
                {"params": predictor.parameters(), "lr": cfg.tent_lr * 10},
            ])
            tta_obj = tent.JEPATTA(
                model, opt, aug_tf, predictor,
                momentum=cfg.jepa_momentum, loss_fn=cfg.jepa_loss,
                use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                steps=cfg.tent_steps)
            print(f"[{method_label}] {len(bn_params)} BN params | "
                  f"Predictor: {sum(p.numel() for p in predictor.parameters())} | "
                  f"EMA: {cfg.jepa_momentum}")

        elif cfg.tta_method == "jepa_con":
            emb_dim = 512
            predictor = tent.PredictorMLP(
                emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
            if warm_predictor_state is not None:
                predictor.load_state_dict(warm_predictor_state)
                print(f"[JEPA_CON] Loaded warm predictor")
            opt = torch.optim.Adam([
                {"params": bn_params, "lr": cfg.tent_lr},
                {"params": predictor.parameters(), "lr": cfg.tent_lr * 10},
            ])
            tta_obj = tent.JEPAContrastive(
                model, opt, aug_tf, predictor,
                con_lambda=cfg.jepa_con_lambda,
                con_temp=cfg.contrastive_temp,
                jepa_lambda=cfg.jepa_train_lambda,
                momentum=cfg.jepa_momentum, loss_fn=cfg.jepa_loss,
                use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                steps=cfg.tent_steps)
            print(f"[JEPA_CON] {len(bn_params)} BN params | "
                  f"Predictor: {sum(p.numel() for p in predictor.parameters())} | "
                  f"λ_con={cfg.jepa_con_lambda} λ_jepa={cfg.jepa_train_lambda}")

    for di, (dn, _, slist) in enumerate(dom_seq):
        if cfg.reset_tta:
            model.load_state_dict(deepcopy(pre_adapt_state))
            model = (tent.configure_model_safe(model) if cfg.safe_bn
                     else tent.configure_model(model))
            bn_params, _ = tent.collect_params(model)
            aug_tf = tent.get_tta_augmentation(cfg.img_size)

            if cfg.tta_method == "contrastive":
                opt = torch.optim.Adam(bn_params, lr=cfg.tent_lr)
                tta_obj = tent.Contrastive(
                    model, opt, aug_tf,
                    contrastive_lambda=cfg.contrastive_lambda,
                    contrastive_temp=cfg.contrastive_temp,
                    use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                    steps=cfg.tent_steps)

            elif cfg.tta_method in ("jepa", "jepa_joint"):
                emb_dim = 512
                predictor = tent.PredictorMLP(
                    emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
                if warm_predictor_state is not None:
                    predictor.load_state_dict(warm_predictor_state)
                opt = torch.optim.Adam([
                    {"params": bn_params, "lr": cfg.tent_lr},
                    {"params": predictor.parameters(),
                     "lr": cfg.tent_lr * 10},
                ])
                tta_obj = tent.JEPATTA(
                    model, opt, aug_tf, predictor,
                    momentum=cfg.jepa_momentum, loss_fn=cfg.jepa_loss,
                    use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                    steps=cfg.tent_steps)

            elif cfg.tta_method == "jepa_con":
                emb_dim = 512
                predictor = tent.PredictorMLP(
                    emb_dim, cfg.jepa_pred_dim, emb_dim).to(cfg.device)
                if warm_predictor_state is not None:
                    predictor.load_state_dict(warm_predictor_state)
                opt = torch.optim.Adam([
                    {"params": bn_params, "lr": cfg.tent_lr},
                    {"params": predictor.parameters(),
                     "lr": cfg.tent_lr * 10},
                ])
                tta_obj = tent.JEPAContrastive(
                    model, opt, aug_tf, predictor,
                    con_lambda=cfg.jepa_con_lambda,
                    con_temp=cfg.contrastive_temp,
                    jepa_lambda=cfg.jepa_train_lambda,
                    momentum=cfg.jepa_momentum, loss_fn=cfg.jepa_loss,
                    use_fft=cfg.use_fft_aug, fft_beta=cfg.fft_beta,
                    steps=cfg.tent_steps)

        print(f"\n  ┌── [{di+1}/{len(dom_seq)}] {dn} "
              f"({[s for s,_,_ in slist]})")

        t0 = time.time()
        tb = sum(len(l) for _, l, _ in slist)

        if cfg.tta_method == "contrastive":
            print(f"  │ {'bat':>5} │{'spec':>6} │{'con':>6} │{'total':>6}")
        elif cfg.tta_method == "jepa_con":
            print(f"  │ {'bat':>5} │{'spec':>6} │{'con':>6} │"
                  f"{'jepa':>6} │{'sim':>6} │{'total':>6}")
        else:
            print(f"  │ {'bat':>5} │{'spec':>6} │{'loss':>6} │"
                  f"{'sim':>6} │{'p_std':>6} │{'t_std':>6}")

        gb = 0
        for sn, ld, _ in slist:
            for imgs, _ in ld:
                _, info = tta_obj(imgs.to(cfg.device))
                if gb < 5 or gb % 50 == 0 or gb == tb - 1:
                    if cfg.tta_method == "contrastive":
                        print(f"  │ {gb:5d} │{sn:>6s} │"
                              f"{info['con']:6.3f} │{info['total']:6.3f}")
                    elif cfg.tta_method == "jepa_con":
                        print(f"  │ {gb:5d} │{sn:>6s} │"
                              f"{info['con']:6.3f} │{info['jepa']:6.3f} │"
                              f"{info['sim']:6.3f} │{info['total']:6.3f}")
                    else:
                        print(f"  │ {gb:5d} │{sn:>6s} │"
                              f"{info['loss']:6.3f} │{info['sim']:6.3f} │"
                              f"{info['p_std']:6.3f} │{info['t_std']:6.3f}")
                gb += 1

        print(f"  │ Adapt: {time.time()-t0:.1f}s")

        # Evaluate this domain
        model.eval()
        for sn, ld, ds in slist:
            gallery_idx, probe_idx = split_gallery_probe(
                ds, cfg.gallery_ratio, cfg.seed)
            all_idx = list(range(len(ds)))
            feats, labels = extract_embeddings(
                model, ds, all_idx, cfg.batch_size, cfg.device,
                cfg.num_workers)
            feats_t = feats.to(cfg.device)
            ver = evaluate_verification(feats_t, labels,
                                         gallery_idx, probe_idx)
            post_tta[sn] = ver
            b = baseline.get(sn, {})
            de = b.get("eer", 0) - ver["eer"]
            dr = ver["rank1"] - b.get("rank1", 0)
            print(f"  │ {sn}: EER {b.get('eer',-1):.2f}→{ver['eer']:.2f}% "
                  f"({'↓' if de > 0 else '↑'}{abs(de):.2f}) | "
                  f"R1 {b.get('rank1',-1):.2f}→{ver['rank1']:.2f}% "
                  f"({'↑' if dr > 0 else '↓'}{abs(dr):.2f})")

        if cfg.reset_tta:
            print(f"  └── Reset for next domain")
        else:
            print(f"  └── Continuing (no reset)")
            model.train()

    # ── Final comparison ──
    mode_str = "episodic" if cfg.reset_tta else "continual"
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON ({method_label}, {mode_str})")
    print(f"  Train IDs: {n_train_cls} | Test IDs: {n_test_cls}")
    print(f"  Source: {cfg.train_spectrums} | Target: "
          f"{[s for s,_,_ in test_loaders]}")
    print(f"{'='*80}")
    print(f"\n  {'Spectrum':<10} {'Pre EER':>9} {'Pre R1':>8} "
          f"{'Post EER':>9} {'Post R1':>8} {'ΔEER':>8} {'ΔR1':>8}")
    print(f"  {'─'*65}")

    for sn in post_tta:
        b = baseline.get(sn, {}); n = post_tta[sn]
        de = b.get("eer", 0) - n["eer"]
        dr = n["rank1"] - b.get("rank1", 0)
        print(f"  {sn:<10} {b.get('eer',-1):>8.2f}% "
              f"{b.get('rank1',-1):>7.2f}% "
              f"{n['eer']:>8.2f}% {n['rank1']:>7.2f}% "
              f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
              f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")

    def _m(d, k):
        return np.mean([r[k] for r in d.values()]) if d else -1
    be = _m(baseline, 'eer'); br = _m(baseline, 'rank1')
    te = _m(post_tta, 'eer'); tr = _m(post_tta, 'rank1')
    de = be - te; dr = tr - br
    print(f"  {'─'*65}")
    print(f"  {'MEAN':<10} {be:>8.2f}% {br:>7.2f}% "
          f"{te:>8.2f}% {tr:>7.2f}% "
          f"{'↓' if de > 0 else '↑'}{abs(de):>6.2f}% "
          f"{'↑' if dr > 0 else '↓'}{abs(dr):>6.2f}%")

    save_data = {
        "tta_method": cfg.tta_method, "mode": mode_str,
        "baseline": {k: dict(v) for k, v in baseline.items()},
        "post_tta": {k: dict(v) for k, v in post_tta.items()},
        "n_train_ids": n_train_cls, "n_test_ids": n_test_cls,
        "train_spectrums": cfg.train_spectrums,
    }
    p = os.path.join(cfg.output_dir,
                     f"casia_{cfg.tta_method}_seed{cfg.seed}.json")
    with open(p, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {p}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = get_cfg()
    set_seed(cfg.seed)
    if cfg.dataset == "casia_ms":
        adapt_casia_ms(cfg)
    else:
        raise ValueError(f"Dataset '{cfg.dataset}' — use contrastive/jepa "
                         f"with casia_ms")
