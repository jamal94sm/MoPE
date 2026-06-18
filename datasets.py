"""
datasets.py — Dataset loaders for TENT TTA.

Handles:
  - ImageNet-C (classification): 15 corruptions × 5 severities
  - CASIA-MS (verification): 6 spectrums, gallery/probe split, EER + Rank-1
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from collections import defaultdict
from sklearn.metrics import roc_curve

from config import IMAGENET_C_CORRUPTIONS, CASIA_MS_SPECTRUMS


# ══════════════════════════════════════════════════════════════
#  ImageNet-C
# ══════════════════════════════════════════════════════════════

class ImageNetCDataset(Dataset):
    """Single corruption type from ImageNet-C at a given severity."""

    def __init__(self, root, corruption, severity=5, transform=None):
        self.root = os.path.join(root, corruption, str(severity))
        self.transform = transform
        self._folder = datasets.ImageFolder(self.root, transform=None)
        self.samples = self._folder.samples
        self.targets = self._folder.targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_imagenet_c_loaders(data_dir, severity=5, batch_size=64,
                            num_workers=4, img_size=224,
                            corruptions=None, norm_mean=None, norm_std=None):
    """
    Returns list of (corruption_name, DataLoader).
    """
    if corruptions:
        valid = set(IMAGENET_C_CORRUPTIONS)
        for c in corruptions:
            if c not in valid:
                raise ValueError(f"Unknown corruption '{c}'")
        run_corruptions = corruptions
    else:
        run_corruptions = IMAGENET_C_CORRUPTIONS

    if norm_mean is None:
        norm_mean = [0.5, 0.5, 0.5]
    if norm_std is None:
        norm_std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    loaders = []
    for corruption in run_corruptions:
        ds = ImageNetCDataset(data_dir, corruption, severity, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
        loaders.append((corruption, loader))
    return loaders


# ══════════════════════════════════════════════════════════════
#  CASIA Multi-Spectral Palmprint
# ══════════════════════════════════════════════════════════════

def _parse_casia_filename(fname):
    """Parse {subjectID}_{handSide}_{spectrum}_{iteration}.jpg"""
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) < 4:
        return None, None, None
    return f"{parts[0]}_{parts[1]}", parts[2], parts[3]


def build_global_identity_map(data_dir):
    """
    Scan ALL files in data_dir and build a consistent identity → index mapping.
    This ensures the same identity gets the same label across all spectrums.

    Returns:
        identity_to_idx: dict {identity_str: int}
        spectrum_files: dict {spectrum: [(path, identity_str), ...]}
    """
    spectrum_files = defaultdict(list)
    all_identities = set()

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        identity, spectrum, _ = _parse_casia_filename(fname)
        if identity is None:
            continue
        all_identities.add(identity)
        spectrum_files[spectrum].append(
            (os.path.join(data_dir, fname), identity))

    identity_to_idx = {ident: idx
                       for idx, ident in enumerate(sorted(all_identities))}

    return identity_to_idx, dict(spectrum_files)


class CASIAMSDataset(Dataset):
    """CASIA-MS dataset for a single spectrum with global identity labels."""

    def __init__(self, file_list, identity_to_idx, transform=None):
        """
        file_list: [(path, identity_str), ...]
        identity_to_idx: global mapping {identity_str: int}
        """
        self.transform = transform
        self.samples = [(path, identity_to_idx[ident])
                        for path, ident in file_list
                        if ident in identity_to_idx]
        self.identity_to_idx = identity_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_identities(self):
        labels = set(label for _, label in self.samples)
        return len(labels)


class CASIAMSTrainDataset(Dataset):
    """
    Combined multi-spectrum dataset for ArcFace training.
    Includes augmentation.
    """
    def __init__(self, file_lists, identity_to_idx, img_size=112):
        """
        file_lists: dict {spectrum: [(path, identity_str), ...]}
        """
        self.identity_to_idx = identity_to_idx
        self.samples = []
        for spectrum, files in file_lists.items():
            for path, ident in files:
                if ident in identity_to_idx:
                    self.samples.append((path, identity_to_idx[ident]))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0, contrast=0.05),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0),
                                             ratio=(1.0, 1.0)),
                transforms.RandomPerspective(distortion_scale=0.1, p=1),
                transforms.RandomRotation(8),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    @property
    def num_identities(self):
        return len(set(label for _, label in self.samples))


def get_casia_ms_train_test(data_dir, train_spectrums, batch_size=64,
                             num_workers=4, img_size=112):
    """
    Build train and test datasets/loaders for CASIA-MS.

    All identities appear in BOTH train and test.
    Train spectrums: used for ArcFace head training (with augmentation).
    Test spectrums: everything else (for evaluation + TENT).

    Returns:
        identity_to_idx: global identity mapping
        num_identities: total identity count
        train_loader: DataLoader for training (combined train spectrums)
        test_loaders: list of (spectrum_name, DataLoader, CASIAMSDataset)
    """
    identity_to_idx, spectrum_files = build_global_identity_map(data_dir)
    num_identities = len(identity_to_idx)

    all_spectrums = sorted(spectrum_files.keys())
    test_spectrums = [s for s in all_spectrums if s not in train_spectrums]

    print(f"[CASIA-MS] {num_identities} identities across "
          f"{len(all_spectrums)} spectrums")
    print(f"  Train spectrums: {train_spectrums}")
    print(f"  Test spectrums:  {test_spectrums}")

    # ── Train dataset (combined train spectrums, with augmentation) ──
    train_files = {s: spectrum_files[s] for s in train_spectrums
                   if s in spectrum_files}
    train_ds = CASIAMSTrainDataset(train_files, identity_to_idx, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)

    total_train = len(train_ds)
    for s in train_spectrums:
        n = len(spectrum_files.get(s, []))
        print(f"    {s}: {n} samples")
    print(f"    Total train: {total_train}")

    # ── Test datasets (per-spectrum, no augmentation) ──
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_loaders = []
    for spectrum in test_spectrums:
        if spectrum not in spectrum_files:
            print(f"  [WARN] No files for test spectrum '{spectrum}'")
            continue
        ds = CASIAMSDataset(spectrum_files[spectrum], identity_to_idx,
                            transform=eval_transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
        test_loaders.append((spectrum, loader, ds))
        print(f"    {spectrum}: {len(ds)} samples, "
              f"{ds.num_identities} IDs in this spectrum")

    return identity_to_idx, num_identities, train_loader, test_loaders


# For backward compat — single-spectrum loaders
def get_casia_ms_loaders(data_dir, batch_size=64, num_workers=4,
                          img_size=112, spectrums=None):
    """Returns list of (spectrum_name, DataLoader, CASIAMSDataset)."""
    identity_to_idx, spectrum_files = build_global_identity_map(data_dir)
    run_spectrums = spectrums if spectrums else sorted(spectrum_files.keys())

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    loaders = []
    for spectrum in run_spectrums:
        if spectrum not in spectrum_files:
            continue
        ds = CASIAMSDataset(spectrum_files[spectrum], identity_to_idx,
                            transform=eval_transform)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
        loaders.append((spectrum, loader, ds))
    return loaders


# ══════════════════════════════════════════════════════════════
#  Verification Evaluation (EER + Rank-1)
# ══════════════════════════════════════════════════════════════

def split_gallery_probe(dataset, gallery_ratio=0.1, seed=2025):
    """
    Closed-set split: all identities in both gallery and probe.
    gallery_ratio of each identity's samples → gallery, rest → probe.

    Returns: gallery_indices, probe_indices
    """
    rng = np.random.RandomState(seed)
    label_to_indices = defaultdict(list)
    for i, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(i)

    gallery_indices = []
    probe_indices = []

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        rng.shuffle(indices)
        n_gallery = max(1, int(len(indices) * gallery_ratio))
        gallery_indices.extend(indices[:n_gallery])
        probe_indices.extend(indices[n_gallery:])

    return gallery_indices, probe_indices


@torch.no_grad()
def extract_embeddings(model, dataset, indices, batch_size, device,
                        num_workers=4):
    """
    Extract embeddings for given sample indices.
    model must have .get_embeddings(x) or be called directly.
    """
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_feats = []
    all_labels = []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        if hasattr(model, 'get_embeddings'):
            feats = model.get_embeddings(imgs)
        elif hasattr(model, 'forward_features'):
            feats = model.forward_features(imgs)
            if feats.dim() == 3:
                feats = feats[:, 0]
        else:
            feats = model(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labs)

    return torch.cat(all_feats, 0), np.concatenate([l.numpy() for l in all_labels])


def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate from genuine/impostor similarity scores."""
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return -1.0
    labels = np.concatenate([np.ones(len(genuine_scores)),
                              np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2 * 100  # percentage


def evaluate_verification(features, labels, gallery_idx, probe_idx):
    """
    Compute EER and Rank-1 from extracted features.
    Returns dict with eer (%), rank1 (%).
    """
    gallery_feats = F.normalize(features[gallery_idx], dim=-1)
    gallery_labels = labels[gallery_idx]
    probe_feats = F.normalize(features[probe_idx], dim=-1)
    probe_labels = labels[probe_idx]

    # Similarity matrix
    sim = (probe_feats @ gallery_feats.T).cpu().numpy()

    # Rank-1
    top1_pred = gallery_labels[sim.argmax(axis=1)]
    rank1 = (top1_pred == probe_labels).mean() * 100

    # EER
    genuine = []
    impostor = []
    for i in range(len(probe_labels)):
        same = gallery_labels == probe_labels[i]
        diff = ~same
        if same.sum() > 0:
            genuine.append(sim[i, same].max())
        if diff.sum() > 0:
            impostor.append(sim[i, diff].max())

    eer = compute_eer(np.array(genuine), np.array(impostor))

    return {"eer": eer, "rank1": rank1,
            "n_gallery": len(gallery_idx), "n_probe": len(probe_idx)}
