# %% [markdown]
# # EoMT-S with Skip Connections — Fine-tuning on COCO Panoptic
# 
# This notebook fine-tunes the EoMT-S model (DINOv2 backbone) with custom
# inter-block skip connections on COCO Panoptic Segmentation.
#
# **Features:**
# - Auto-download of COCO 2017 dataset
# - Two modes: pretrained EoMT-S weights or from-scratch initialization
# - Checkpointing every N epochs + best model saving
# - Resume from any checkpoint
# - Live train/val loss plots
# - PQ (Panoptic Quality) evaluation on validation set
# - Optimized for Kaggle P100 (16GB VRAM)

# %% [markdown]
# ## 1. Setup & Install Dependencies

# %%
import subprocess, sys, os

def install_deps():
    """Install all required packages."""
    packages = [
        "gitignore_parser==0.1.12",
        "jsonargparse[signatures]==4.38",
        "matplotlib>=3.9,<3.11",
        "timm==1.0.15",
        "wandb==0.19.10",
        "lightning==2.5.1.post0",
        "transformers==4.56.1",
        "scipy>=1.15",
        "torchmetrics==1.7.1",
        "pycocotools==2.0.8",
        "fvcore==0.1.5.post20221221",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

install_deps()

# %%
import shutil

REPO_URL = "https://github.com/SergKurchev/eomt_skip_connections.git"
REPO_DIR = "/kaggle/working/eomt_skip_connections"

if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
    print(f"Cloned repo to {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Imports ready.")

# %% [markdown]
# ## 2. Configuration

# %%
CONFIG = {
    # --- Mode ---
    "mode": "pretrained",          # "pretrained" (load EoMT-S weights) or "scratch"
    "use_skip_connections": True,  # Set to True for skip connections, False for original EoMT architecture
    "resume_from": None,           # Path to checkpoint .pt file, or None

    # --- Paths ---
    # The awsaf49 Kaggle dataset lacks PANOPTIC annotations.
    # We must read images from Kaggle Input, but download annotations to Working Dir.
    "coco_images_dir": "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017",
    "coco_annotations_dir": "/kaggle/working/coco2017_annotations",
    "output_dir": "/kaggle/working/checkpoints",

    # --- Training ---
    "num_epochs": 24,
    "save_every": 100,             # Save checkpoint every N epochs
    "batch_size": 4,               # P100 16GB: 4 for 640x640 (try, fallback to 2 if OOM)
    "gradient_accumulation_steps": 4,  # Effective batch = 4*4 = 16 (as in paper)
    "lr": 1e-4,
    "weight_decay": 0.05,
    "llrd": 0.8,                   # Layer-wise LR decay factor
    "poly_power": 0.9,
    "warmup_steps": (500, 1000),   # (non-ViT warmup, ViT warmup)
    "llrd_l2_enabled": True,       # Enable LLRD for L2 blocks
    "lr_mult": 1.0,                # LR multiplier for specific blocks
    "gradient_clip_max_norm": 0.01,

    # --- Model ---
    "backbone_name": "vit_small_patch14_reg4_dinov2",
    "num_q": 200,
    "num_blocks": 3,
    "img_size": (640, 640),
    "num_classes": 133,
    "masked_attn_enabled": True,
    "attn_mask_annealing_enabled": True,
    "attn_mask_annealing_start_steps": [29564, 73910, 118256],
    "attn_mask_annealing_end_steps": [59128, 103474, 147820],

    # --- Loss ---
    "no_object_coefficient": 0.1,
    "mask_coefficient": 5.0,
    "dice_coefficient": 5.0,
    "class_coefficient": 2.0,
    "num_points": 12544,
    "oversample_ratio": 3.0,
    "importance_sample_ratio": 0.75,

    # --- Eval ---
    "mask_thresh": 0.8,
    "overlap_thresh": 0.8,

    # --- Data ---
    "num_workers": 4,
    "mixed_precision": True,
    "scale_range": (0.1, 2.0),

    # --- Pretrained weights URL ---
    "pretrained_url": "https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/main/pytorch_model.bin",

    # --- COCO stuff classes (from config yaml) ---
    "stuff_classes": list(range(80, 133)),
}

# Adjust output directory based on skip connections to keep results separate
_config_suffix = "_skip" if CONFIG["use_skip_connections"] else "_original"
CONFIG["output_dir"] = os.path.join(CONFIG["output_dir"], f"run{_config_suffix}")

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("Config loaded. Mode:", CONFIG["mode"], "| Skip Connections:", CONFIG["use_skip_connections"])
print("Checkpoints will be saved to:", CONFIG["output_dir"])

# %% [markdown]
# ## 3. Download COCO 2017 Dataset

# %%
import urllib.request
import zipfile
from pathlib import Path

COCO_URLS = {
    # We only download the panoptic and regular annotations (~1.1 GB).
    # Images (19 GB) are read directly from the Kaggle Input dataset.
    "panoptic_annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def _download_with_progress(url, dest_path):
    """Download a file with progress indicator, rate-limiting prints to prevent IOPub errors."""
    import time
    
    # Store state in a mutable object so the inner function can modify it
    state = {"last_print_time": 0}

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            current_time = time.time()
            # Only print if 2 seconds have passed since the last print, or if it's the very first block
            if current_time - state["last_print_time"] > 2.0 or state["last_print_time"] == 0:
                pct = min(100, downloaded * 100 / total_size)
                gb_down = downloaded / 1e9
                gb_total = total_size / 1e9
                print(f"\r    {pct:5.1f}% | {gb_down:.2f} / {gb_total:.2f} GB", end="", flush=True)
                state["last_print_time"] = current_time

    urllib.request.urlretrieve(url, str(dest_path), reporthook=_reporthook)
    print()  # newline after progress


def download_and_extract(url, dest_dir, filename):
    """Download a zip file, extract it, then DELETE the zip to save disk space."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / filename

    # Check if already extracted
    extract_marker = dest_dir / f".{filename}.extracted"
    if extract_marker.exists():
        print(f"  ✓ Already downloaded & extracted: {filename}")
        return

    # Download
    if zip_path.exists():
        print(f"  Zip already downloaded: {filename}")
    else:
        print(f"  Downloading: {filename} ...")
        _download_with_progress(url, zip_path)
        print(f"  Downloaded: {filename} ({zip_path.stat().st_size / 1e9:.1f} GB)")

    # Extract
    print(f"  Extracting: {filename} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))

    # For panoptic annotations, there's a nested zip
    if "panoptic" in filename:
        nested_zips = list(dest_dir.glob("annotations/panoptic_*.zip"))
        for nz in nested_zips:
            print(f"  Extracting nested: {nz.name} ...")
            with zipfile.ZipFile(str(nz), "r") as zf:
                zf.extractall(str(dest_dir))
            # Delete nested zip too
            nz.unlink()
            print(f"  Deleted nested zip: {nz.name}")

    # DELETE the zip file to free disk space (critical on Kaggle ~57 GB limit)
    zip_path.unlink()
    print(f"  🗑️ Deleted zip to free space: {filename}")

    extract_marker.touch()
    print(f"  ✓ Extracted: {filename}")

def download_coco_annotations(annotations_root):
    """Download ONLY COCO 2017 annotations (to save disk space). Images must come from Kaggle Input."""
    print("=" * 60)
    print("Downloading COCO 2017 Annotations (~1.1 GB total)")
    print("=" * 60)
    for filename, url in COCO_URLS.items():
        download_and_extract(url, annotations_root, filename)
    print("Annotations download complete!")
    print(f"Annotations root: {annotations_root}")

download_coco_annotations(CONFIG["coco_annotations_dir"])

# %% [markdown]
# ## 4. Dataset

# %%
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as TF
from PIL import Image

# Import transforms from the repo
from datasets.transforms import Transforms

# COCO panoptic class mapping (same as datasets/coco_panoptic.py)
CLASS_MAPPING = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 92: 80, 93: 81,
    95: 82, 100: 83, 107: 84, 109: 85, 112: 86, 118: 87, 119: 88,
    122: 89, 125: 90, 128: 91, 130: 92, 133: 93, 138: 94, 141: 95,
    144: 96, 145: 97, 147: 98, 148: 99, 149: 100, 151: 101, 154: 102,
    155: 103, 156: 104, 159: 105, 161: 106, 166: 107, 168: 108,
    171: 109, 175: 110, 176: 111, 177: 112, 178: 113, 180: 114,
    181: 115, 184: 116, 185: 117, 186: 118, 187: 119, 188: 120,
    189: 121, 190: 122, 191: 123, 192: 124, 193: 125, 194: 126,
    195: 127, 196: 128, 197: 129, 198: 130, 199: 131, 200: 132,
}


class COCOPanopticDirect(Dataset):
    """COCO Panoptic dataset that reads images from Kaggle input and annotations from Kaggle working."""

    def __init__(self, coco_images_dir, coco_annotations_dir, split="train", img_size=(640, 640),
                 transforms=None, scale_range=(0.1, 2.0)):
        super().__init__()
        self.images_root = Path(coco_images_dir)
        self.ann_root = Path(coco_annotations_dir)
        self.split = split
        self.img_size = img_size
        self.transforms_fn = transforms

        # Load annotations
        ann_file = self.ann_root / "annotations" / f"panoptic_{split}2017.json"
        with open(ann_file, "r") as f:
            ann_data = json.load(f)

        # Build image_id -> filename mapping
        self.id_to_filename = {img["id"]: img["file_name"] for img in ann_data["images"]}

        # Build annotations indexed by image_id
        self.annotations = {}
        for ann in ann_data["annotations"]:
            img_id = ann["image_id"]
            self.annotations[img_id] = ann

        # Filter to images that have annotations
        self.image_ids = [
            img_id for img_id in self.id_to_filename
            if img_id in self.annotations
        ]

        self.img_dir = self.images_root / f"{split}2017"
        self.mask_dir = self.ann_root / f"panoptic_{split}2017"

        print(f"  [{split}] Loaded {len(self.image_ids)} images with panoptic annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_filename = self.id_to_filename[img_id]
        ann = self.annotations[img_id]

        # Load image
        img_path = self.img_dir / img_filename
        img = Image.open(img_path).convert("RGB")
        img = tv_tensors.Image(img)

        # Load panoptic mask
        mask_filename = ann["file_name"]
        mask_path = self.mask_dir / mask_filename
        mask_img = np.array(Image.open(mask_path))
        # Panoptic mask: RGB -> segment_id
        panoptic_map = mask_img[:, :, 0].astype(np.int64) + \
                       mask_img[:, :, 1].astype(np.int64) * 256 + \
                       mask_img[:, :, 2].astype(np.int64) * 256 * 256
        panoptic_map = torch.from_numpy(panoptic_map)

        # Parse segments
        masks_list, labels_list, is_crowd_list = [], [], []
        for seg_info in ann["segments_info"]:
            cat_id = seg_info["category_id"]
            if cat_id not in CLASS_MAPPING:
                continue
            seg_id = seg_info["id"]
            mask = (panoptic_map == seg_id)
            if not mask.any():
                continue
            masks_list.append(mask)
            labels_list.append(CLASS_MAPPING[cat_id])
            is_crowd_list.append(bool(seg_info["iscrowd"]))

        if len(masks_list) == 0:
            # Fallback: return a dummy target, will be filtered by transforms
            masks_list = [torch.zeros(img.shape[-2:], dtype=torch.bool)]
            labels_list = [0]
            is_crowd_list = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks_list)),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd_list, dtype=torch.bool),
        }

        if self.transforms_fn is not None:
            img, target = self.transforms_fn(img, target)

        return img, target


def build_dataloaders(config):
    """Build train and val dataloaders."""
    train_transforms = Transforms(
        img_size=config["img_size"],
        color_jitter_enabled=False,
        scale_range=config["scale_range"],
    )

    train_dataset = COCOPanopticDirect(
        coco_images_dir=config["coco_images_dir"],
        coco_annotations_dir=config["coco_annotations_dir"],
        split="train",
        img_size=config["img_size"],
        transforms=train_transforms,
    )

    val_dataset = COCOPanopticDirect(
        coco_images_dir=config["coco_images_dir"],
        coco_annotations_dir=config["coco_annotations_dir"],
        split="val",
        img_size=config["img_size"],
        transforms=None,  # No augmentation for val
    )

    def train_collate(batch):
        imgs, targets = [], []
        for img, target in batch:
            imgs.append(img)
            targets.append(target)
        return torch.stack(imgs), targets

    def val_collate(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=train_collate,
        persistent_workers=config["num_workers"] > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Val uses variable-size images
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=val_collate,
        persistent_workers=config["num_workers"] > 0,
    )

    return train_loader, val_loader


print("Dataset classes defined.")

# %% [markdown]
# ## 5. Build Model

# %%
import torch.nn as nn
import torch.nn.functional as F_nn
from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_loss import MaskClassificationLoss


def download_pretrained_weights(url, dest_path):
    """Download pretrained model weights."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"Weights already downloaded: {dest_path}")
        return str(dest_path)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained weights from {url} ...")
    urllib.request.urlretrieve(url, str(dest_path))
    print(f"Downloaded: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return str(dest_path)


def build_model(config):
    """Build EoMT model with skip connections."""
    # Build ViT encoder
    encoder = ViT(
        img_size=config["img_size"],
        backbone_name=config["backbone_name"],
    )

    # Build EoMT
    model = EoMT(
        encoder=encoder,
        num_classes=config["num_classes"],
        num_q=config["num_q"],
        num_blocks=config["num_blocks"],
        masked_attn_enabled=config["masked_attn_enabled"],
        use_skip_connections=config["use_skip_connections"],
    )

    return model


def load_pretrained_weights(model, config):
    """Load pretrained EoMT-S weights, handling missing keys from skip connections."""
    weights_path = download_pretrained_weights(
        config["pretrained_url"],
        Path(config["output_dir"]) / "pretrained" / "pytorch_model.bin",
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Remove criterion weights if present
    state_dict = {k: v for k, v in state_dict.items() if "criterion.empty_weight" not in k}

    # The pretrained weights are saved with "network." prefix from LightningModule
    # Check if keys have "network." prefix
    has_network_prefix = any(k.startswith("network.") for k in state_dict.keys())

    if has_network_prefix:
        # Strip "network." prefix
        state_dict = {
            k.replace("network.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("network.")
        }

    # Load with strict=False to handle skip connection/new params
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"Missing keys (expected for skip connections and new layers):")
        for k in incompatible.missing_keys:
            print(f"  - {k}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys:")
        for k in incompatible.unexpected_keys:
            print(f"  - {k}")

    print(f"Loaded pretrained weights. {len(state_dict)} keys loaded.")
    return model


print("Model building functions defined.")

# %% [markdown]
# ## 6. Optimizer & Scheduler

# %%
from torch.optim import AdamW


def build_optimizer(model, config, total_steps):
    """Build AdamW optimizer with layer-wise learning rate decay (LLRD)."""
    encoder_param_names = {
        n for n, _ in model.encoder.backbone.named_parameters()
    }

    backbone_param_groups = []
    other_param_groups = []
    backbone_blocks = len(model.encoder.backbone.blocks)
    block_i = backbone_blocks
    
    # Blocks that correspond to EoMT blocks (refinement blocks)
    l2_blocks = list(range(backbone_blocks - config["num_blocks"], backbone_blocks))

    for name, param in reversed(list(model.named_parameters())):
        if not param.requires_grad:
            continue

        lr = config["lr"]

        # Check if this is an encoder backbone parameter
        stripped_name = name.replace("encoder.backbone.", "")
        if stripped_name in encoder_param_names:
            name_parts = name.split(".")

            is_block = False
            for i_part, key in enumerate(name_parts):
                if key == "blocks":
                    block_i = int(name_parts[i_part + 1])
                    is_block = True

            if is_block or block_i == 0:
                lr *= config["llrd"] ** (backbone_blocks - 1 - block_i)
            
            # Additional logic for refinement blocks (match lightning_module.py)
            if (
                is_block
                and (block_i in l2_blocks)
                and ((not config["llrd_l2_enabled"]) or (config["lr_mult"] != 1.0))
            ):
                lr = config["lr"]

            if "backbone.norm" in name:
                lr = config["lr"]

            backbone_param_groups.append(
                {"params": [param], "lr": lr, "name": name}
            )
        else:
            other_param_groups.append(
                {"params": [param], "lr": config["lr"], "name": name}
            )

    param_groups = backbone_param_groups + other_param_groups
    optimizer = AdamW(param_groups, weight_decay=config["weight_decay"])

    print(f"Optimizer: AdamW with {len(backbone_param_groups)} backbone param groups, "
          f"{len(other_param_groups)} other param groups")

    return optimizer, len(backbone_param_groups)


class PolyWarmupScheduler:
    """Two-stage warmup + polynomial decay LR scheduler.

    Mimics TwoStageWarmupPolySchedule from the repo.
    """

    def __init__(self, optimizer, num_backbone_params, warmup_steps, total_steps, poly_power):
        self.optimizer = optimizer
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        step = self._step
        non_vit_warmup, vit_warmup = self.warmup_steps

        for i, (pg, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            if i >= self.num_backbone_params:
                # Non-backbone params
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                # Backbone params
                if step < non_vit_warmup:
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    lr = base_lr * ((step - non_vit_warmup) / vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup - vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power

            pg["lr"] = max(lr, 1e-8)  # Clamp to avoid zero LR

    def state_dict(self):
        return {"step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict):
        self._step = state_dict["step"]
        self.base_lrs = state_dict["base_lrs"]


print("Optimizer & scheduler defined.")

def attn_mask_annealing(model, config, global_step):
    """Update attention mask probabilities for each block.
    
    Mimics LightningModule.mask_annealing logic.
    """
    if not config["attn_mask_annealing_enabled"]:
        return

    start_steps = config["attn_mask_annealing_start_steps"]
    end_steps = config["attn_mask_annealing_end_steps"]
    
    # attn_mask_probs is a buffer in model (EoMT class)
    # It should have length equal to config["num_blocks"]
    probs = torch.ones(config["num_blocks"], device=next(model.parameters()).device)
    
    for i in range(config["num_blocks"]):
        if global_step < start_steps[i]:
            probs[i] = 1.0
        elif global_step > end_steps[i]:
            probs[i] = 0.0
        else:
            # Polynomial decay (Paper & Repo match)
            total_anneal_steps = end_steps[i] - start_steps[i]
            if total_anneal_steps > 0:
                progress = (global_step - start_steps[i]) / total_anneal_steps
                probs[i] = (1.0 - progress) ** config["poly_power"]
            else:
                probs[i] = 0.0
    
    # Update buffer
    model.attn_mask_probs.copy_(probs)

# %% [markdown]
# ## 7. Evaluation (PQ)

# %%
from torchmetrics.detection import PanopticQuality


def build_pq_metric(config, device):
    """Build PanopticQuality metric."""
    thing_classes = [i for i in range(config["num_classes"]) if i not in config["stuff_classes"]]
    stuff_classes_with_void = config["stuff_classes"] + [config["num_classes"]]
    metric = PanopticQuality(
        things=thing_classes,
        stuffs=stuff_classes_with_void,
        return_sq_and_rq=True,
        return_per_class=True,
    ).to(device)
    return metric


def to_per_pixel_preds_panoptic(mask_logits_list, class_logits, num_classes,
                                 stuff_classes, mask_thresh, overlap_thresh):
    """Convert mask/class logits to per-pixel panoptic predictions."""
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    preds_list = []

    for i in range(len(mask_logits_list)):
        preds = -torch.ones(
            (*mask_logits_list[i].shape[-2:], 2),
            dtype=torch.long,
            device=class_logits.device,
        )
        preds[:, :, 0] = num_classes

        keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
        if not keep.any():
            preds_list.append(preds)
            continue

        masks = mask_logits_list[i].sigmoid()
        segments = -torch.ones(
            *masks.shape[-2:], dtype=torch.long, device=class_logits.device,
        )

        mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
        stuff_segment_ids, segment_id = {}, 0
        segment_and_class_ids = []

        for k, class_id in enumerate(classes[i][keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = orig_mask & new_mask

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum().item()
            if (orig_area == 0 or new_area == 0 or final_area == 0
                    or new_area / orig_area < overlap_thresh):
                continue

            if class_id in stuff_classes:
                if class_id in stuff_segment_ids:
                    segments[final_mask] = stuff_segment_ids[class_id]
                    continue
                else:
                    stuff_segment_ids[class_id] = segment_id

            segments[final_mask] = segment_id
            segment_and_class_ids.append((segment_id, class_id))
            segment_id += 1

        for seg_id, cls_id in segment_and_class_ids:
            segment_mask = segments == seg_id
            preds[:, :, 0] = torch.where(segment_mask, cls_id, preds[:, :, 0])
            preds[:, :, 1] = torch.where(segment_mask, seg_id, preds[:, :, 1])

        preds_list.append(preds)

    return preds_list


def to_per_pixel_targets_panoptic(targets):
    """Convert targets to per-pixel format for PQ metric."""
    per_pixel_targets = []
    for target in targets:
        masks = target["masks"]
        labels = target["labels"]
        per_pixel_target = -torch.ones(
            (*masks.shape[-2:], 2),
            dtype=labels.dtype,
            device=labels.device,
        )

        for i, mask in enumerate(masks):
            per_pixel_target[:, :, 0] = torch.where(
                mask, labels[i], per_pixel_target[:, :, 0]
            )
            per_pixel_target[:, :, 1] = torch.where(
                mask,
                torch.tensor(i, device=masks.device),
                per_pixel_target[:, :, 1],
            )
        per_pixel_targets.append(per_pixel_target)

    return per_pixel_targets


@torch.no_grad()
def validate(model, val_loader, criterion, pq_metric, config, device):
    """Run validation: compute loss and PQ."""
    model.eval()
    pq_metric.reset()

    total_loss = 0.0
    num_batches = 0
    img_size = config["img_size"]
    stuff_classes = config["stuff_classes"]
    num_classes = config["num_classes"]
    mask_thresh = config["mask_thresh"]
    overlap_thresh = config["overlap_thresh"]
    
    vis_data = []

    from tqdm.auto import tqdm
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for imgs_tuple, targets_tuple in pbar:
        imgs_list = list(imgs_tuple)
        targets_list = list(targets_tuple)

        # Resize and pad images for inference
        processed_imgs = []
        img_sizes = []
        for img in imgs_list:
            img = img.to(device, dtype=torch.float32)
            img_sizes.append(img.shape[-2:])
            h, w = img.shape[-2:]
            factor = min(img_size[0] / h, img_size[1] / w)
            new_h, new_w = round(h * factor), round(w * factor)
            img_resized = F_nn.interpolate(
                img[None].float(), size=(new_h, new_w), mode="bilinear", align_corners=False
            )[0]
            # Pad
            pad_h = max(0, img_size[0] - new_h)
            pad_w = max(0, img_size[1] - new_w)
            img_padded = F_nn.pad(img_resized, (0, pad_w, 0, pad_h))
            processed_imgs.append(img_padded)

        imgs_batch = torch.stack(processed_imgs)

        with torch.amp.autocast("cuda", enabled=config["mixed_precision"]):
            mask_logits_per_block, class_logits_per_block = model(imgs_batch / 255.0)

        # Use only the last block output for validation metrics
        mask_logits = mask_logits_per_block[-1]
        class_logits = class_logits_per_block[-1]

        # Compute loss
        targets_device = []
        for t in targets_list:
            td = {
                "masks": t["masks"].to(device, dtype=torch.float32),
                "labels": t["labels"].to(device),
            }
            targets_device.append(td)

        try:
            with torch.amp.autocast("cuda", enabled=config["mixed_precision"]):
                losses = criterion(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=targets_device,
                )
            loss_val = sum(
                loss * (config["mask_coefficient"] if "mask" in k
                        else config["dice_coefficient"] if "dice" in k
                        else config["class_coefficient"])
                for k, loss in losses.items()
            )
            total_loss += loss_val.item()
            num_batches += 1
        except Exception:
            pass  # Skip problematic batches

        # PQ metric
        mask_logits_interp = F_nn.interpolate(mask_logits, img_size, mode="bilinear")

        # Revert resize and pad
        reverted_logits = []
        for i_img in range(len(imgs_list)):
            h, w = img_sizes[i_img]
            factor = min(img_size[0] / h, img_size[1] / w)
            scaled_h, scaled_w = round(h * factor), round(w * factor)
            logits_i = mask_logits_interp[i_img][:, :scaled_h, :scaled_w]
            logits_i = F_nn.interpolate(
                logits_i[None], size=(h, w), mode="bilinear", align_corners=False
            )[0]
            reverted_logits.append(logits_i)

        preds = to_per_pixel_preds_panoptic(
            reverted_logits, class_logits, num_classes,
            stuff_classes, mask_thresh, overlap_thresh,
        )

        gt_targets = to_per_pixel_targets_panoptic(
            [{"masks": t["masks"].to(device), "labels": t["labels"].to(device)}
             for t in targets_list]
        )

        for i_img, (pred, gt) in enumerate(zip(preds, gt_targets)):
            try:
                pq_metric.update(pred[None].to(gt.device, dtype=gt.dtype), gt[None])
            except Exception:
                pass
                
            if len(vis_data) < 4:
                # Save first 4 validation images for visualization
                vis_data.append((
                    imgs_list[i_img].cpu(),
                    gt[:, :, 0].cpu(),
                    pred[:, :, 0].cpu()
                ))

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)

    try:
        # Compute then drop the void class (last element) and take mean
        # pq_result shape is (num_classes, 3) representing [PQ, SQ, RQ]
        pq_result = pq_metric.compute()[:-1]
        pq_value = pq_result[:, 0].mean().item()
        sq_value = pq_result[:, 1].mean().item()
        rq_value = pq_result[:, 2].mean().item()
    except Exception:
        pq_value, sq_value, rq_value = 0.0, 0.0, 0.0

    model.train()
    return avg_loss, pq_value, sq_value, rq_value, vis_data


print("Validation functions defined.")

# %% [markdown]
# ## 8. Checkpointing & Plotting

# %%
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, best_pq,
                    train_losses, val_losses, pq_scores, config, filename="checkpoint.pt"):
    """Save a training checkpoint."""
    path = Path(config["output_dir"]) / filename
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_pq": best_pq,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "pq_scores": pq_scores,
        "config": config,
    }, str(path))
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt["epoch"],
        ckpt.get("global_step", 0),
        ckpt["best_val_loss"],
        ckpt.get("best_pq", 0.0),
        ckpt["train_losses"],
        ckpt["val_losses"],
        ckpt.get("pq_scores", []),
    )


def plot_losses(train_losses, val_losses, pq_scores, epoch, config):
    """Plot training progress with live update."""
    clear_output(wait=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # --- Train loss (smoothed) ---
    ax1 = axes[0]
    if train_losses:
        # Smooth with exponential moving average
        smoothed = []
        alpha = 0.98
        s = train_losses[0]
        for v in train_losses:
            s = alpha * s + (1 - alpha) * v
            smoothed.append(s)
        ax1.plot(smoothed, color="#2196F3", linewidth=0.8, alpha=0.9, label="Train Loss (EMA)")
        ax1.plot(train_losses, color="#2196F3", linewidth=0.2, alpha=0.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Train Loss — Epoch {epoch}/{config['num_epochs']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Val loss ---
    ax2 = axes[1]
    if val_losses:
        epochs_axis = list(range(1, len(val_losses) + 1))
        ax2.plot(epochs_axis, val_losses, "o-", color="#FF5722", linewidth=1.5,
                 markersize=3, label="Val Loss")
        best_idx = np.argmin(val_losses)
        ax2.plot(best_idx + 1, val_losses[best_idx], "*", color="#4CAF50",
                 markersize=15, label=f"Best: {val_losses[best_idx]:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- PQ ---
    ax3 = axes[2]
    if pq_scores:
        epochs_axis = list(range(1, len(pq_scores) + 1))
        ax3.plot(epochs_axis, [pq * 100 for pq in pq_scores], "s-", color="#9C27B0",
                 linewidth=1.5, markersize=3, label="PQ")
        best_idx = np.argmax(pq_scores)
        ax3.plot(best_idx + 1, pq_scores[best_idx] * 100, "*", color="#4CAF50",
                 markersize=15, label=f"Best: {pq_scores[best_idx]*100:.1f}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PQ (%)")
    ax3.set_title("Panoptic Quality")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(config["output_dir"]) / "training_progress.png", dpi=100)
    plt.show()

def plot_visualizations(vis_data, epoch, config):
    """Plot validation image segmentations (GT vs Pred)."""
    if not vis_data:
        return
        
    num_imgs = len(vis_data)
    fig, axes = plt.subplots(2, num_imgs, figsize=(6 * num_imgs, 10))
    if num_imgs == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i, (img, gt, pred) in enumerate(vis_data):
        # Image is (3, H, W) uint8 tensor usually, or float
        if img.dtype.is_floating_point:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
            
        gt_np = gt.numpy()
        pred_np = pred.numpy()
        
        # Row 0: Ground Truth
        ax_gt = axes[0, i]
        ax_gt.imshow(img_np)
        mask_gt = np.ma.masked_where(gt_np == config["num_classes"], gt_np)
        ax_gt.imshow(mask_gt, cmap="tab20", alpha=0.6, vmin=0, vmax=config["num_classes"])
        ax_gt.axis("off")
        if i == 0: ax_gt.set_title(f"Ground Truth (Epoch {epoch})", fontsize=14, loc="left", pad=10)
        
        # Row 1: Prediction
        ax_pred = axes[1, i]
        ax_pred.imshow(img_np)
        mask_pred = np.ma.masked_where(pred_np == config["num_classes"], pred_np)
        ax_pred.imshow(mask_pred, cmap="tab20", alpha=0.6, vmin=0, vmax=config["num_classes"])
        ax_pred.axis("off")
        if i == 0: ax_pred.set_title("Prediction", fontsize=14, loc="left", pad=10)
        
    plt.tight_layout()
    plt.savefig(Path(config["output_dir"]) / f"vis_epoch_{epoch}.png", dpi=100)
    plt.show()

print("Checkpoint & plotting functions defined.")

# %% [markdown]
# ## 9. Training Loop

# %%
def train(config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build data ---
    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")

    # --- Build model ---
    print("\nBuilding model...")
    model = build_model(config)

    if config["mode"] == "pretrained":
        model = load_pretrained_weights(model, config)
    else:
        print("Training from scratch (random initialization).")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # --- Build loss ---
    criterion = MaskClassificationLoss(
        num_points=config["num_points"],
        oversample_ratio=config["oversample_ratio"],
        importance_sample_ratio=config["importance_sample_ratio"],
        mask_coefficient=config["mask_coefficient"],
        dice_coefficient=config["dice_coefficient"],
        class_coefficient=config["class_coefficient"],
        num_labels=config["num_classes"],
        no_object_coefficient=config["no_object_coefficient"],
    ).to(device)

    # --- Build optimizer & scheduler ---
    steps_per_epoch = len(train_loader) // config["gradient_accumulation_steps"]
    total_steps = steps_per_epoch * config["num_epochs"]
    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")

    optimizer, num_backbone_params = build_optimizer(model, config, total_steps)
    scheduler = PolyWarmupScheduler(
        optimizer, num_backbone_params,
        config["warmup_steps"], total_steps, config["poly_power"],
    )

    # --- Build PQ metric ---
    pq_metric = build_pq_metric(config, device)

    # --- Resume from checkpoint ---
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_pq = 0.0
    train_losses = []
    val_losses = []
    pq_scores = []

    if config["resume_from"] is not None:
        print(f"\nResuming from checkpoint: {config['resume_from']}")
        start_epoch, global_step, best_val_loss, best_pq, train_losses, val_losses, pq_scores = \
            load_checkpoint(config["resume_from"], model, optimizer, scheduler, device)
        print(f"  Resumed from epoch {start_epoch}, step {global_step}, best val loss: {best_val_loss:.4f}, best PQ: {best_pq*100:.1f}")

    # --- Mixed precision scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Starting training: epochs {start_epoch+1} -> {config['num_epochs']}")
    print(f"{'='*60}\n")

    model.train()
    accum_steps = config["gradient_accumulation_steps"]

    from tqdm.auto import tqdm
    
    interrupted = False

    try:
        for epoch in range(start_epoch, config["num_epochs"]):
            epoch_train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
            for batch_idx, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(device)
                targets_device = [
                    {
                        "masks": t["masks"].to(device, dtype=torch.float32),
                        "labels": t["labels"].to(device),
                    }
                    for t in targets
                ]

                # Update attention mask probabilities (annealing)
                attn_mask_annealing(model, config, global_step)

                with torch.amp.autocast("cuda", enabled=config["mixed_precision"]):
                    mask_logits_per_block, class_logits_per_block = model(imgs / 255.0)

                    # Compute loss for all blocks (as in original training)
                    total_loss = None
                    for mask_logits, class_logits in zip(
                        mask_logits_per_block, class_logits_per_block
                    ):
                        losses = criterion(
                            masks_queries_logits=mask_logits,
                            class_queries_logits=class_logits,
                            targets=targets_device,
                        )
                        block_loss = sum(
                            loss * (config["mask_coefficient"] if "mask" in k
                                    else config["dice_coefficient"] if "dice" in k
                                    else config["class_coefficient"])
                            for k, loss in losses.items()
                        )
                        if total_loss is None:
                            total_loss = block_loss
                        else:
                            total_loss = total_loss + block_loss

                    total_loss = total_loss / accum_steps

                scaler.scale(total_loss).backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_max_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                step_loss = total_loss.item() * accum_steps
                train_losses.append(step_loss)
                epoch_train_loss += step_loss

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    lr_current = optimizer.param_groups[-1]["lr"]
                    pbar.set_postfix(Loss=f"{step_loss:.4f}", LR=f"{lr_current:.2e}")

        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
        print(f"\nEpoch {epoch+1} — Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        print(f"  Running validation...")
        val_loss, pq_value, sq_value, rq_value, vis_data = validate(
            model, val_loader, criterion, pq_metric, config, device
        )
        val_losses.append(val_loss)
        pq_scores.append(pq_value)
        print(f"  Val Loss: {val_loss:.4f} | PQ: {pq_value*100:.1f}% | SQ: {sq_value*100:.1f}% | RQ: {rq_value*100:.1f}%")

        # --- Save best model ---
        is_best = False
        if pq_value > best_pq:
            best_pq = pq_value
            is_best = True
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                best_val_loss, best_pq, train_losses, val_losses, pq_scores,
                config, filename="best_model.pt",
            )
            print(f"  ★ New best PQ: {best_pq*100:.1f}%")

        # --- Periodic checkpoint save ---
        if (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                best_val_loss, best_pq, train_losses, val_losses, pq_scores,
                config, filename=f"checkpoint_epoch_{epoch+1}.pt",
            )

        # --- Always save latest (for resume) EVERY epoch ---
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, global_step,
            best_val_loss, best_pq, train_losses, val_losses, pq_scores,
            config, filename="latest.pt",
        )

        # --- Live plot ---
        plot_losses(train_losses, val_losses, pq_scores, epoch + 1, config)
        plot_visualizations(vis_data, epoch + 1, config)

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (KeyboardInterrupt).")
        interrupted = True
    except Exception as e:
        print(f"\n[!] Training aborted due to error: {e}")
        interrupted = True
        raise e
    finally:
        if interrupted and 'epoch' in locals():
            print("\nSaving emergency checkpoint...")
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                best_val_loss, best_pq, train_losses, val_losses, pq_scores,
                config, filename="interrupted.pt",
            )

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best PQ: {best_pq*100:.1f}%")
    print(f"Checkpoints saved to: {config['output_dir']}")
    print(f"{'='*60}")

# %% [markdown]
# ## 10. Run Training

# %%
# =============================================
# MODIFY CONFIG HERE BEFORE RUNNING
# =============================================

# To resume from a checkpoint:
# CONFIG["resume_from"] = "/kaggle/working/checkpoints/latest.pt"

# To train from scratch (no pretrained weights):
# CONFIG["mode"] = "scratch"

# To change the COCO paths (if your Kaggle dataset varies):
# CONFIG["coco_images_dir"] = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
# CONFIG["coco_annotations_dir"] = "/kaggle/working/coco2017_annotations"

# =============================================

train(CONFIG)
