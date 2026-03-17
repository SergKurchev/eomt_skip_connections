# %% [markdown]
# # Validation of Original EoMT-S 2x (No Skip Connections) on COCO Panoptic
# 
# This notebook validates the **original** EoMT-S model (from the official
# `tue-mps/eomt` repo, **without** skip connections) using the same pretrained
# EoMT-S 2x weights on the COCO 2017 panoptic val set.
#
# **Purpose:** Provide a baseline comparison for our skip-connection variant.

# %% [markdown]
# ## 1. Setup & Install Dependencies

# %%
import subprocess, sys, os

def install_deps():
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

# install_deps()

# %%
# Clone the ORIGINAL EoMT repo (no skip connections)
ORIG_REPO_URL = "https://github.com/tue-mps/eomt.git"
ORIG_REPO_DIR = "/kaggle/working/eomt_original"

# if not os.path.exists(ORIG_REPO_DIR):
#     subprocess.check_call(["git", "clone", ORIG_REPO_URL, ORIG_REPO_DIR])
#     print(f"Cloned ORIGINAL EoMT repo to {ORIG_REPO_DIR}")
# else:
#     print(f"Original repo already exists at {ORIG_REPO_DIR}")

if ORIG_REPO_DIR not in sys.path:
    sys.path.insert(0, ORIG_REPO_DIR)

print("Imports ready.")

# %% [markdown]
# ## 2. Configuration

# %%
CONFIG = {
    # --- Paths ---
    "coco_images_dir": "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017",
    "coco_annotations_dir": "/kaggle/working/coco2017_annotations",
    "output_dir": "/kaggle/working/validation_results",

    # --- Model ---
    "backbone_name": "vit_small_patch14_reg4_dinov2",
    "num_q": 200,
    "num_blocks": 3,
    "img_size": (640, 640),
    "num_classes": 133,
    "masked_attn_enabled": True,

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
    "mixed_precision": True,

    # --- Data ---
    "num_workers": 4,
    # Set to an integer (e.g. 50) to run a quick smoke-test on N images.
    # Set to None to run the full 5000-image validation.
    "num_val_samples": 50,

    # --- Pretrained weights URL ---
    "pretrained_url": "https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/main/pytorch_model.bin",

    # --- COCO stuff classes ---
    "stuff_classes": list(range(80, 133)),
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("Config loaded.")

# %% [markdown]
# ## 3. Download COCO Panoptic Annotations

# %%
import urllib.request
import zipfile
import json
import time
from pathlib import Path

COCO_URLS = {
    "panoptic_annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download_with_progress(url, dest_path):
    state = {"last_print_time": 0}

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            current_time = time.time()
            if current_time - state["last_print_time"] > 2.0 or state["last_print_time"] == 0:
                pct = min(100, downloaded * 100 / total_size)
                gb_down = downloaded / 1e9
                gb_total = total_size / 1e9
                print(f"\r    {pct:5.1f}% | {gb_down:.2f} / {gb_total:.2f} GB", end="", flush=True)
                state["last_print_time"] = current_time

    urllib.request.urlretrieve(url, str(dest_path), reporthook=_reporthook)
    print()


def download_and_extract(url, dest_dir, filename):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / filename
    extract_marker = dest_dir / f".{filename}.extracted"

    if extract_marker.exists():
        print(f"  Already extracted: {filename}")
        return

    print(f"  Downloading: {filename} ...")
    _download_with_progress(url, zip_path)
    print(f"  Downloaded: {filename} ({zip_path.stat().st_size / 1e9:.1f} GB)")

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

    zip_path.unlink()
    print(f"  🗑️ Deleted zip to free space: {filename}")
    extract_marker.touch()
    print(f"  ✓ Extracted: {filename}")


def download_coco_annotations(annotations_root):
    print("=" * 60)
    print("Downloading COCO 2017 Annotations (~1.1 GB total)")
    print("=" * 60)
    for filename, url in COCO_URLS.items():
        download_and_extract(url, annotations_root, filename)
    print("Annotations download complete!")

download_coco_annotations(CONFIG["coco_annotations_dir"])

# %% [markdown]
# ## 4. Dataset

# %%
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from PIL import Image

# COCO category ID -> contiguous class index (same mapping as training notebook)
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
    """COCO Panoptic dataset for validation."""

    def __init__(self, coco_images_dir, coco_annotations_dir, split="val", img_size=(640, 640)):
        super().__init__()
        self.images_root = Path(coco_images_dir)
        self.ann_root = Path(coco_annotations_dir)
        self.split = split
        self.img_size = img_size

        ann_file = self.ann_root / "annotations" / f"panoptic_{split}2017.json"
        with open(ann_file, "r") as f:
            ann_data = json.load(f)

        self.id_to_filename = {img["id"]: img["file_name"] for img in ann_data["images"]}
        self.annotations = {}
        for ann in ann_data["annotations"]:
            self.annotations[ann["image_id"]] = ann

        self.image_ids = [img_id for img_id in self.id_to_filename if img_id in self.annotations]
        self.img_dir = self.images_root / f"{split}2017"
        self.mask_dir = self.ann_root / f"panoptic_{split}2017"
        print(f"  [{split}] Loaded {len(self.image_ids)} images with panoptic annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_filename = self.id_to_filename[img_id]
        ann = self.annotations[img_id]

        img = Image.open(self.img_dir / img_filename).convert("RGB")
        img = tv_tensors.Image(img)

        mask_img = np.array(Image.open(self.mask_dir / ann["file_name"]))
        panoptic_map = (mask_img[:, :, 0].astype(np.int64) +
                        mask_img[:, :, 1].astype(np.int64) * 256 +
                        mask_img[:, :, 2].astype(np.int64) * 256 * 256)
        panoptic_map = torch.from_numpy(panoptic_map)

        masks_list, labels_list = [], []
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

        if len(masks_list) == 0:
            masks_list = [torch.zeros(img.shape[-2:], dtype=torch.bool)]
            labels_list = [0]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks_list)),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        return img, target


print("Dataset defined.")

# %% [markdown]
# ## 5. Build Original Model & Load Weights

# %%
import torch.nn as nn
import torch.nn.functional as F_nn
from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_loss import MaskClassificationLoss


def build_original_model(config):
    """Build the ORIGINAL EoMT model (no skip connections)."""
    encoder = ViT(
        img_size=config["img_size"],
        backbone_name=config["backbone_name"],
    )
    model = EoMT(
        encoder=encoder,
        num_classes=config["num_classes"],
        num_q=config["num_q"],
        num_blocks=config["num_blocks"],
        masked_attn_enabled=config["masked_attn_enabled"],
        use_skip_connections=False,
    )
    return model


def load_pretrained_weights(model, config):
    """Load pretrained EoMT-S 2x weights."""
    weights_path = Path(config["output_dir"]) / "pretrained" / "pytorch_model.bin"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"Downloading pretrained weights...")
        urllib.request.urlretrieve(config["pretrained_url"], str(weights_path))
        print(f"Downloaded: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Weights already downloaded: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k: v for k, v in state_dict.items() if "criterion.empty_weight" not in k}

    has_network_prefix = any(k.startswith("network.") for k in state_dict.keys())
    if has_network_prefix:
        state_dict = {
            k.replace("network.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("network.")
        }

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {incompatible.unexpected_keys}")

    loaded = len(state_dict) - len(incompatible.unexpected_keys)
    print(f"Loaded pretrained weights. {loaded} keys matched.")
    return model


print("Model building functions defined.")

# %% [markdown]
# ## 6. Post-processing & Metrics

# %%
from torchmetrics.detection import PanopticQuality


def build_pq_metric(config, device):
    """Build PanopticQuality metric."""
    thing_ids = sorted([i for i in range(config["num_classes"]) if i not in config["stuff_classes"]])
    stuff_ids = sorted(config["stuff_classes"] + [config["num_classes"]])  # +void
    
    return PanopticQuality(
        things=thing_ids,
        stuffs=stuff_ids,
        return_sq_and_rq=True,
        return_per_class=True,
    ).to(device)


def to_per_pixel_preds_panoptic(mask_logits_list, class_logits, num_classes,
                                 stuff_classes, mask_thresh, overlap_thresh):
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    preds_list = []

    for i in range(len(mask_logits_list)):
        preds = -torch.ones(
            (*mask_logits_list[i].shape[-2:], 2), dtype=torch.long,
            device=class_logits.device,
        )
        preds[:, :, 0] = num_classes

        keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
        if not keep.any():
            preds_list.append(preds)
            continue

        masks = mask_logits_list[i].sigmoid()
        segments = -torch.ones(*masks.shape[-2:], dtype=torch.long, device=class_logits.device)
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
            if (orig_area == 0 or new_area == 0 or final_area == 0 or new_area / orig_area < overlap_thresh):
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
    per_pixel_targets = []
    for target in targets:
        masks = target["masks"]
        labels = target["labels"]
        per_pixel_target = -torch.ones(
            (*masks.shape[-2:], 2), dtype=labels.dtype, device=labels.device,
        )
        for i, mask in enumerate(masks):
            per_pixel_target[:, :, 0] = torch.where(mask, labels[i], per_pixel_target[:, :, 0])
            per_pixel_target[:, :, 1] = torch.where(
                mask, torch.tensor(i, device=masks.device), per_pixel_target[:, :, 1],
            )
        per_pixel_targets.append(per_pixel_target)
    return per_pixel_targets


print("Post-processing & metrics defined.")

# %% [markdown]
# ## 7. Full Validation

# %%
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


@torch.no_grad()
def validate_full(model, val_loader, criterion, pq_metric, config, device):
    """Run full validation: loss and PQ."""
    model.eval()
    pq_metric.reset()

    total_loss = 0.0
    num_batches = 0
    img_size = config["img_size"]
    stuff_classes = config["stuff_classes"]
    num_classes = config["num_classes"]
    mask_thresh = config["mask_thresh"]
    overlap_thresh = config["overlap_thresh"]
    num_val_samples = config.get("num_val_samples", None)
    samples_processed = 0

    vis_data = []
    pbar = tqdm(val_loader, desc="Validating original EoMT-S 2x")
    for imgs_tuple, targets_tuple in pbar:
        imgs_list = list(imgs_tuple)
        targets_list = list(targets_tuple)

        # Resize and pad
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
            pad_h = max(0, img_size[0] - new_h)
            pad_w = max(0, img_size[1] - new_w)
            img_padded = F_nn.pad(img_resized, (0, pad_w, 0, pad_h))
            processed_imgs.append(img_padded)

        imgs_batch = torch.stack(processed_imgs)

        with torch.amp.autocast("cuda", enabled=config["mixed_precision"]):
            mask_logits_per_block, class_logits_per_block = model(imgs_batch / 255.0)

        mask_logits = mask_logits_per_block[-1]
        class_logits = class_logits_per_block[-1]

        # Loss
        targets_device = [
            {
                "masks": t["masks"].to(device, dtype=torch.float32),
                "labels": t["labels"].to(device),
            }
            for t in targets_list
        ]

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
            pass

        # PQ metric
        mask_logits_interp = F_nn.interpolate(mask_logits, img_size, mode="bilinear")
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
            except Exception as e:
                batch_idx = num_batches
                print(f"  [PQ UPDATE ERROR] batch={batch_idx}, img_in_batch={i_img}: {e}")

            if len(vis_data) < 4:
                vis_data.append((
                    imgs_list[i_img].cpu(),
                    gt[:, :, 0].cpu(),
                    pred[:, :, 0].cpu(),
                ))

        samples_processed += len(imgs_list)
        if num_val_samples is not None and samples_processed >= num_val_samples:
            print(f"  Stopping early at {samples_processed} samples (num_val_samples={num_val_samples})")
            break

        if num_batches % 500 == 0 and num_batches > 0:
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    # Compute metrics
    avg_loss = total_loss / max(num_batches, 1)

    try:
        # Original code logic: metric.compute()[:-1] -> drops the void class
        # then computes mean across all other classes
        pq_result = pq_metric.compute()[:-1]
        pq_value = pq_result[:, 0].mean().item()
        sq_value = pq_result[:, 1].mean().item()
        rq_value = pq_result[:, 2].mean().item()
    except Exception as e:
        print(f"PQ compute error: {e}")
        import traceback; traceback.print_exc()
        pq_value = 0.0
        sq_value = 0.0
        rq_value = 0.0

    return avg_loss, pq_value, sq_value, rq_value, vis_data


print("Validation function defined.")

# %% [markdown]
# ## 8. Run Validation

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Build dataset ---
print("\nBuilding val dataset...")
val_dataset = COCOPanopticDirect(
    coco_images_dir=CONFIG["coco_images_dir"],
    coco_annotations_dir=CONFIG["coco_annotations_dir"],
    split="val",
    img_size=CONFIG["img_size"],
)

def val_collate(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
    collate_fn=val_collate,
    persistent_workers=CONFIG["num_workers"] > 0,
)
print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")

# --- Build model ---
print("\nBuilding ORIGINAL EoMT-S model (no skip connections)...")
model = build_original_model(CONFIG)
model = load_pretrained_weights(model, CONFIG)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")

# --- Build loss ---
criterion = MaskClassificationLoss(
    num_points=CONFIG["num_points"],
    oversample_ratio=CONFIG["oversample_ratio"],
    importance_sample_ratio=CONFIG["importance_sample_ratio"],
    mask_coefficient=CONFIG["mask_coefficient"],
    dice_coefficient=CONFIG["dice_coefficient"],
    class_coefficient=CONFIG["class_coefficient"],
    num_labels=CONFIG["num_classes"],
    no_object_coefficient=CONFIG["no_object_coefficient"],
).to(device)

# --- Build PQ metric (with SQ and RQ) ---
pq_metric = build_pq_metric(CONFIG, device)

# --- Run validation ---
print("\n" + "=" * 60)
print("Validating ORIGINAL EoMT-S 2x on COCO 2017 val set")
print("=" * 60 + "\n")

avg_loss, pq_value, sq_value, rq_value, vis_data = validate_full(
    model, val_loader, criterion, pq_metric, CONFIG, device
)

# %% [markdown]
# ## 9. Results

# %%
print("\n" + "=" * 60)
print("VALIDATION RESULTS — Original EoMT-S 2x (No Skip Connections)")
print("=" * 60)
print(f"  Val Loss:         {avg_loss:.4f}")
print(f"  Panoptic Quality: {pq_value * 100:.1f}%")
print(f"  Segmentation Q:   {sq_value * 100:.1f}%")
print(f"  Recognition Q:    {rq_value * 100:.1f}%")
print(f"  Total Parameters: {total_params:,}")
print(f"  Expected PQ (paper): 46.7%")
print("=" * 60)

# Save results to file
results = {
    "model": "EoMT-S 2x (Original, No Skip Connections)",
    "val_loss": avg_loss,
    "pq": pq_value,
    "total_params": total_params,
    "paper_pq": 0.467,
}

results_path = Path(CONFIG["output_dir"]) / "original_eomt_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_path}")

# %% [markdown]
# ## 10. Visualization: GT vs Prediction

# %%
def plot_visualizations(vis_data, config, title="Original EoMT-S 2x"):
    """Plot validation image segmentations (GT vs Pred)."""
    if not vis_data:
        print("No visualization data collected!")
        return

    num_imgs = len(vis_data)
    fig, axes = plt.subplots(2, num_imgs, figsize=(6 * num_imgs, 10))
    if num_imgs == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, (img, gt, pred) in enumerate(vis_data):
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
        if i == 0:
            ax_gt.set_title(f"Ground Truth — {title}", fontsize=14, loc="left", pad=10)

        # Row 1: Prediction
        ax_pred = axes[1, i]
        ax_pred.imshow(img_np)
        mask_pred = np.ma.masked_where(pred_np == config["num_classes"], pred_np)
        ax_pred.imshow(mask_pred, cmap="tab20", alpha=0.6, vmin=0, vmax=config["num_classes"])
        ax_pred.axis("off")
        if i == 0:
            ax_pred.set_title("Prediction", fontsize=14, loc="left", pad=10)

    plt.suptitle(f"Panoptic Segmentation — {title}\nPQ: {pq_value*100:.1f}% | SQ: {sq_value*100:.1f}% | RQ: {rq_value*100:.1f}%",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(Path(config["output_dir"]) / "vis_original_eomt.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to: {Path(config['output_dir']) / 'vis_original_eomt.png'}")


plot_visualizations(vis_data, CONFIG)
