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


class CASIAMSDataset(Dataset):
    """CASIA Multi-Spectral palmprint dataset for a single spectrum."""

    def __init__(self, root, spectrum, transform=None):
        self.root = root
        self.spectrum = spectrum
        self.transform = transform
        self.samples = []
        self.identities = []
        self.identity_to_idx = {}

        identity_samples = defaultdict(list)
        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            identity, spec, _ = _parse_casia_filename(fname)
            if identity is None or spec != spectrum:
                continue
            identity_samples[identity].append(os.path.join(root, fname))

        self.identities = sorted(identity_samples.keys())
        self.identity_to_idx = {ident: idx
                                for idx, ident in enumerate(self.identities)}

        for ident in self.identities:
            for fpath in identity_samples[ident]:
                self.samples.append((fpath, self.identity_to_idx[ident]))

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
        return len(self.identities)


def get_casia_ms_loaders(data_dir, batch_size=64, num_workers=4,
                          img_size=112, spectrums=None):
    """
    Returns list of (spectrum_name, DataLoader, CASIAMSDataset).
    ArcFace convention: 112×112, mean=0.5, std=0.5.
    """
    run_spectrums = spectrums if spectrums else CASIA_MS_SPECTRUMS

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    loaders = []
    for spectrum in run_spectrums:
        ds = CASIAMSDataset(data_dir, spectrum, transform=transform)
        if len(ds) == 0:
            print(f"[WARN] No samples for spectrum '{spectrum}' in {data_dir}")
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
