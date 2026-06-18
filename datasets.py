"""
datasets.py — Dataset loaders for all CTTA benchmarks.
Handles: ImageNet-C, CIFAR-100-C, ImageNet+/++ (CRS), ACDC.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import timm

from config import (IMAGENET_C_CORRUPTIONS, CIFAR100_C_CORRUPTIONS,
                    CRS_DOMAINS)


# ─── ImageNet-C ───────────────────────────────────────────────────────

class ImageNetCDataset(Dataset):
    """Single corruption type from ImageNet-C at a given severity."""

    def __init__(self, root, corruption, severity=5, transform=None):
        self.root = os.path.join(root, corruption, str(severity))
        self.transform = transform

        # use torchvision ImageFolder to enumerate samples
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


def get_imagenet_c_sequence(data_dir, severity=5, batch_size=50,
                            num_workers=4, img_size=224, corruptions=None,
                            backbone_name="vit_base_patch16_224"):
    """
    Returns a list of (corruption_name, DataLoader) pairs.
    corruptions: None  → all 15 standard corruptions
                 list  → only the specified corruption(s)
    """
    if corruptions:
        # validate names
        valid = set(IMAGENET_C_CORRUPTIONS)
        for c in corruptions:
            if c not in valid:
                raise ValueError(
                    f"Unknown corruption '{c}'. "
                    f"Valid: {IMAGENET_C_CORRUPTIONS}")
        run_corruptions = corruptions
    else:
        run_corruptions = IMAGENET_C_CORRUPTIONS


    # use timm's recommended transform for this specific checkpoint
    model = timm.create_model(backbone_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    del model  # don't need the model, just the transform

    '''

    # 2. Replicate the baseline's 'Res256Crop224' exactly using Bicubic interpolation:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             #std=[0.229, 0.224, 0.225]),
    ])
    '''
    
    loaders = []
    for corruption in run_corruptions:
        ds = ImageNetCDataset(data_dir, corruption, severity, transform)
        # 3. Change shuffle back to False for stable evaluation tracking
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
        loaders.append((corruption, loader))

    
    return loaders


# ─── CIFAR-100-C ──────────────────────────────────────────────────────

class CIFAR100CDataset(Dataset):
    """Single corruption type from CIFAR-100-C .npy files."""

    def __init__(self, root, corruption, severity=5, transform=None):
        # each .npy file: [50000, 32, 32, 3] uint8 (all 5 severity levels)
        data_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")

        all_data = np.load(data_path)       # [50000, 32, 32, 3]
        all_labels = np.load(labels_path)   # [50000]

        # each severity: 10000 samples
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = all_data[start:end]
        self.labels = all_labels[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar100_c_sequence(data_dir, severity=5, batch_size=50,
                             num_workers=4, img_size=384):
    """Returns list of (corruption_name, DataLoader) for CIFAR-100-C."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    loaders = []
    for corruption in CIFAR100_C_CORRUPTIONS:
        ds = CIFAR100CDataset(data_dir, corruption, severity, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, ## was False
                            num_workers=num_workers, pin_memory=True)
        loaders.append((corruption, loader))

    return loaders


# ─── CRS Benchmark: ImageNet+ / ImageNet++ ────────────────────────────

# Class index mappings for partial-class datasets
# ImageNet-A: 200 classes, ImageNet-R: 200 classes

def _load_class_mapping(dataset_dir):
    """
    Build mapping from folder names to ImageNet 1000-class indices.
    Returns dict: folder_name → imagenet_class_idx
    """
    # Standard ImageNet synset order (from torchvision)
    # We load it from the folder structure assuming folders are named
    # as WordNet IDs (nXXXXXXXX)
    folders = sorted(os.listdir(dataset_dir))
    folders = [f for f in folders if os.path.isdir(os.path.join(dataset_dir, f))]

    # We need to map these to the standard 1000-class indices.
    # For ImageNet-A and ImageNet-R, we load a mapping file or
    # derive it from the folder names.
    return {f: i for i, f in enumerate(folders)}


class ImageNetVariantDataset(Dataset):
    """
    Loader for ImageNet-V2, ImageNet-A, ImageNet-R, ImageNet-Sketch.
    Handles class mapping to the full 1000-class space.
    """

    def __init__(self, root, transform=None, class_mapping=None):
        """
        root: path to dataset folder
        class_mapping: dict mapping folder_name → 1000-class index.
                       If None, uses ImageFolder's default ordering.
        """
        self.transform = transform
        self._folder = datasets.ImageFolder(root, transform=None)
        self.samples = self._folder.samples
        self.class_to_idx = self._folder.class_to_idx

        # build label remapping if needed
        if class_mapping is not None:
            self.label_map = class_mapping
            self._use_mapping = True
        else:
            self.label_map = None
            self._use_mapping = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self._use_mapping:
            # remap from dataset-local index to ImageNet 1000 index
            folder_name = os.path.basename(os.path.dirname(path))
            label = self.label_map.get(folder_name, label)

        return img, label


def _get_crs_dataset_path(data_dir, domain_name):
    """Map domain name to actual directory."""
    paths = {
        "imagenet_v2": os.path.join(data_dir, "imagenetv2-matched-frequency-format-val"),
        "imagenet_a": os.path.join(data_dir, "imagenet-a"),
        "imagenet_r": os.path.join(data_dir, "imagenet-r"),
        "imagenet_sketch": os.path.join(data_dir, "imagenet-sketch"),
    }
    # try alternative names
    for alt in [domain_name, domain_name.replace("_", "-"),
                domain_name.replace("imagenet_", "imagenet-")]:
        candidate = os.path.join(data_dir, alt)
        if os.path.isdir(candidate):
            return candidate

    return paths.get(domain_name, os.path.join(data_dir, domain_name))


def get_crs_sequence(data_dir, num_rounds=3, batch_size=50,
                     num_workers=4, img_size=224, plusplus=False):
    """
    Build the CRS domain sequence: (V2→A→R→S) × num_rounds.
    For ImageNet++, uses non-overlapping subsets per round (simplified:
    we shuffle and split for each dataset).

    Returns list of (domain_name, DataLoader).
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pre-load all datasets
    domain_datasets = {}
    for domain in CRS_DOMAINS:
        path = _get_crs_dataset_path(data_dir, domain)
        if not os.path.isdir(path):
            print(f"[WARNING] Dataset path not found: {path}")
            continue
        domain_datasets[domain] = ImageNetVariantDataset(
            path, transform=transform)

    # build sequence
    sequence = []
    for round_idx in range(num_rounds):
        for domain in CRS_DOMAINS:
            if domain not in domain_datasets:
                continue

            ds = domain_datasets[domain]

            if plusplus and domain != "imagenet_a":
                # non-overlapping subset: split into num_rounds parts
                total = len(ds)
                chunk = total // num_rounds
                start = round_idx * chunk
                end = start + chunk if round_idx < num_rounds - 1 else total
                indices = list(range(start, end))
                subset = torch.utils.data.Subset(ds, indices)
            else:
                subset = ds

            loader = DataLoader(subset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=False)
            name = f"{domain}_R{round_idx+1}"
            sequence.append((name, loader))

    return sequence


# ─── Unified interface ────────────────────────────────────────────────

def get_domain_sequence(cfg):
    """
    Build the domain sequence based on cfg.dataset.
    Returns list of (domain_name, DataLoader).
    """
    if cfg.dataset == "imagenet_c":
        return get_imagenet_c_sequence(
            cfg.data_dir, cfg.severity, cfg.batch_size,
            cfg.num_workers, cfg.img_size, cfg.corruptions)

    elif cfg.dataset == "cifar100_c":
        return get_cifar100_c_sequence(
            cfg.data_dir, cfg.severity, cfg.batch_size,
            cfg.num_workers, cfg.img_size)

    elif cfg.dataset == "imagenet_plus":
        return get_crs_sequence(
            cfg.data_dir, cfg.num_rounds, cfg.batch_size,
            cfg.num_workers, cfg.img_size, plusplus=False)

    elif cfg.dataset == "imagenet_plusplus":
        return get_crs_sequence(
            cfg.data_dir, cfg.num_rounds, cfg.batch_size,
            cfg.num_workers, cfg.img_size, plusplus=True)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
