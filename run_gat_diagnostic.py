"""
GAT View-Consistency Diagnostic Inference Script
=================================================
Runs GAT affordance segmentation on multi-view ShapeNet renders
and saves per-view masks + raw similarity scores for analysis.

Usage:
    python run_gat_diagnostic.py \
        --image_dir path/to/rendered_views/ \
        --affordance grasp \
        --model_file best_8_iou_68.4.pth \
        --output_dir results/

Directory structure expected for --image_dir:
    rendered_views/
        mug_0001_az0.png
        mug_0001_az45.png
        mug_0001_az90.png
        mug_0001_az135.png
        mug_0001_az180.png
        knife_0001_az0.png
        ...

Or pass --image_list for explicit file list.

Prerequisites (run from affordance-learning/ directory):
    1. dinov2_vitb14_pretrain.pth in working directory
       wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
    2. best_8_iou_68.4.pth in working directory
       wget https://huggingface.co/Gen1113/Model_for_Aff-Grasp/resolve/main/best_8_iou_68.4.pth
    3. Custom CUDA ops compiled:
       cd models/dino/ops && python3.10 setup.py build install && cd ../../..
    4. pip install opencv-python tqdm timm peft matplotlib depth_anything (or use precomputed depth)
"""

import os
import sys
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Affordance vocabulary (must match training order) ──────────────────────────
AFF_LIST = ['grasp', 'cut', 'scoop', 'pound', 'support', 'screw', 'contain', 'stick']
AFF2IDX  = {a: i for i, a in enumerate(AFF_LIST)}

# ── Colour palette for multi-affordance overlay (one colour per class) ─────────
PALETTE = [
    [128, 128, 128], [129, 127,  38], [120,  69, 125], [ 53, 125,  34],
    [  0,  11, 123], [118,  20,  12], [122,  81,  25], [241, 134,  51]
]

# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing  (mirrors TestData in ego_video_data.py)
# ──────────────────────────────────────────────────────────────────────────────
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)
DEP_MEAN = (0.5,   0.5,   0.5)
DEP_STD  = (0.5,   0.5,   0.5)

def build_transforms(crop_size: int):
    img_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    dep_tf = transforms.Compose([
        transforms.Resize((crop_size, crop_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEP_MEAN, std=DEP_STD),
    ])
    return img_tf, dep_tf


def load_image(path: str, img_tf) -> tuple:
    """Returns (tensor [1,3,H,W], original PIL image)."""
    pil = Image.open(path).convert('RGB')
    return img_tf(pil).unsqueeze(0), pil


def compute_depth(pil_img: Image.Image, dep_tf, depth_model=None) -> torch.Tensor:
    """
    Compute pseudo-depth via Depth-Anything if available,
    otherwise return a zero tensor (depth-free inference).

    Returns tensor [1, 3, H, H] normalised for GAT.
    """
    if depth_model is not None:
        import numpy as np
        img_np = np.array(pil_img)
        depth_np = depth_model.infer_image(img_np)          # [H, W] float
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_uint8 = (depth_np * 255).astype(np.uint8)
        # replicate to 3 channels (grayscale → RGB) as in training
        depth_rgb = np.stack([depth_uint8]*3, axis=-1)
        depth_pil = Image.fromarray(depth_rgb)
        return dep_tf(depth_pil).unsqueeze(0)
    else:
        # zero depth → model falls back to pure DINOv2 features
        size = 448   # will be overridden by actual input size below
        return None  # handled at call site


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_gat(model_file: str, device: torch.device):
    from models.GAT import Net
    model = Net().to(device)
    model.eval()
    assert os.path.exists(model_file), f"Checkpoint not found: {model_file}"
    ckpt = torch.load(model_file, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"[✓] Loaded GAT from {model_file}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference for a single image
# ──────────────────────────────────────────────────────────────────────────────
def run_single(model, img_tensor, dep_tensor, device, bg_threshold=0.8):
    """
    Returns:
        pred_norm  : [num_aff, H, W]  similarity scores, normalised to [0,1]
        mask       : [H, W] bool      binary mask for queried affordance after thresholding
        raw_sim    : [num_aff, H, W]  raw cosine similarity (before normalisation)
        max_sim    : float            peak similarity across all pixels & affordances
    """
    img_tensor = img_tensor.to(device)
    dep_tensor = dep_tensor.to(device) if dep_tensor is not None else None

    with torch.no_grad():
        pred = model(img_tensor, dep_tensor)          # [1, num_aff, H, W]

    pred = pred.squeeze(0)                            # [num_aff, H, W]
    raw_sim = pred.cpu()

    # Normalise across the entire prediction volume (matches test.py)
    pred_min, pred_max = pred.min(), pred.max()
    pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-10)
    pred_norm = pred_norm.cpu()

    max_sim = pred_norm.max().item()

    return pred_norm, raw_sim, max_sim


# ──────────────────────────────────────────────────────────────────────────────
# Failure classification (Section C of the proposal)
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_LOW_THRESHOLD  = 0.4   # max_sim below this → identity-like failure
BG_THRESHOLD          = 0.8   # per-pixel threshold for binary mask (matches test.py)

def classify_failure(pred_norm, aff_idx, max_sim):
    """
    Returns one of: 'success' | 'identity_failure' | 'affordance_failure'

    identity_failure  : model has no confident response at all (globally low activation)
    affordance_failure: model responds confidently but the peak is not on the target affordance
    """
    if max_sim < GLOBAL_LOW_THRESHOLD:
        return 'identity_failure'

    # Check if the target affordance channel has any above-threshold pixels
    aff_map = pred_norm[aff_idx]
    if aff_map.max().item() < BG_THRESHOLD:
        return 'affordance_failure'

    return 'success'


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────
def overlay_mask(orig_pil, pred_norm, aff_idx, bg_threshold=BG_THRESHOLD, alpha=0.5):
    """Return numpy RGB image with affordance mask overlaid in red."""
    orig_np = np.array(orig_pil.convert('RGB'))
    h, w = orig_np.shape[:2]

    aff_map = pred_norm[aff_idx].numpy()              # [H, W] 0-1
    aff_map_resized = cv2.resize(aff_map, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (aff_map_resized >= bg_threshold).astype(np.uint8)

    overlay = orig_np.copy()
    overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) +
                          np.array([255, 0, 0]) * alpha).astype(np.uint8)

    # Draw heatmap channel as colourised inset
    heatmap = cm.jet(aff_map_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    return overlay, heatmap, mask


def save_result_figure(orig_pil, pred_norm, aff_idx, aff_name,
                       failure_type, max_sim, out_path):
    orig_np = np.array(orig_pil.convert('RGB'))
    overlay, heatmap, mask = overlay_mask(orig_pil, pred_norm, aff_idx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np);    axes[0].set_title('Input');         axes[0].axis('off')
    axes[1].imshow(heatmap);    axes[1].set_title(f'Similarity ({aff_name})'); axes[1].axis('off')
    axes[2].imshow(overlay);    axes[2].set_title(f'{failure_type} | max_sim={max_sim:.2f}'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main diagnostic loop
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # ── validate affordance ────────────────────────────────────────────────────
    assert args.affordance in AFF2IDX, \
        f"Unknown affordance '{args.affordance}'. Choose from {AFF_LIST}"
    aff_idx  = AFF2IDX[args.affordance]
    aff_name = args.affordance
    print(f"[affordance] querying '{aff_name}' (channel {aff_idx})")

    # ── build transforms ───────────────────────────────────────────────────────
    img_tf, dep_tf = build_transforms(args.crop_size)

    # ── optionally load Depth-Anything ─────────────────────────────────────────
    depth_model = None
    if args.use_depth:
        try:
            from depth_anything.dpt import DepthAnything
            depth_model = DepthAnything.from_pretrained('LiheYoung/depth-anything-large-hf').to(device)
            depth_model.eval()
            print("[✓] Depth-Anything loaded")
        except ImportError:
            print("[!] depth_anything not installed — running without depth (pass None to model)")

    # ── load GAT ───────────────────────────────────────────────────────────────
    model = load_gat(args.model_file, device)

    # ── collect image paths ────────────────────────────────────────────────────
    if args.image_list:
        image_paths = [p.strip() for p in open(args.image_list).readlines() if p.strip()]
    else:
        exts = {'.png', '.jpg', '.jpeg'}
        image_paths = sorted([
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        ])
    assert len(image_paths) > 0, "No images found."
    print(f"[images] {len(image_paths)} images to process")

    # ── output directories ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    viz_dir = out_dir / 'viz'
    npy_dir = out_dir / 'raw_scores'
    viz_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # ── per-image results accumulator ─────────────────────────────────────────
    results = []

    for img_path in tqdm(image_paths):
        stem = Path(img_path).stem

        # Load & preprocess RGB
        img_tensor, orig_pil = load_image(img_path, img_tf)

        # Depth: compute or use zeros
        if depth_model is not None:
            dep_tensor = compute_depth(orig_pil, dep_tf, depth_model)
            dep_tensor = dep_tensor.to(device) if dep_tensor is not None else None
        else:
            if args.use_depth:
                # try to find a pre-computed depth file alongside the image
                dep_path = img_path.replace('.png', '_graydepth.png').replace('.jpg', '_graydepth.png')
                if os.path.exists(dep_path):
                    dep_pil  = Image.open(dep_path).convert('RGB')
                    dep_tensor = dep_tf(dep_pil).unsqueeze(0)
                else:
                    print(f"  [!] No depth found for {stem}, using zeros")
                    dep_tensor = torch.zeros(1, 3, args.crop_size, args.crop_size)
            else:
                # Run without depth intentionally
                dep_tensor = torch.zeros(1, 3, args.crop_size, args.crop_size)

        # Run GAT
        pred_norm, raw_sim, max_sim = run_single(model, img_tensor, dep_tensor, device)

        # Classify failure
        failure_type = classify_failure(pred_norm, aff_idx, max_sim)

        # Save raw similarity scores for later quantitative analysis
        np.save(str(npy_dir / f'{stem}_sim.npy'), pred_norm.numpy())

        # Save visualisation
        viz_path = str(viz_dir / f'{stem}_viz.png')
        save_result_figure(orig_pil, pred_norm, aff_idx, aff_name,
                           failure_type, max_sim, viz_path)

        # Per-affordance max similarity scores (useful for identity probe)
        per_aff_max = {aff: float(pred_norm[i].max()) for i, aff in enumerate(AFF_LIST)}

        result = {
            'image'       : img_path,
            'stem'        : stem,
            'affordance'  : aff_name,
            'aff_idx'     : aff_idx,
            'max_sim_global': max_sim,
            'max_sim_target': float(pred_norm[aff_idx].max()),
            'per_aff_max' : per_aff_max,
            'failure_type': failure_type,
        }
        results.append(result)
        tqdm.write(f"  {stem:40s} | max_sim={max_sim:.3f} | target_max={pred_norm[aff_idx].max():.3f} | {failure_type}")

    # ── save summary JSON ──────────────────────────────────────────────────────
    summary_path = out_dir / 'results.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved to {summary_path}")

    # ── print failure breakdown ────────────────────────────────────────────────
    failure_counts = {}
    for r in results:
        ft = r['failure_type']
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    print("\n── Failure Breakdown ──────────────────────────────────")
    for k, v in sorted(failure_counts.items()):
        print(f"  {k:25s}: {v}/{len(results)} ({100*v/len(results):.1f}%)")
    print("────────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT View-Consistency Diagnostic')

    # paths
    parser.add_argument('--model_file',  type=str, required=True,
                        help='Path to best_8_iou_68.4.pth')
    parser.add_argument('--image_dir',   type=str, default=None,
                        help='Directory of rendered view images')
    parser.add_argument('--image_list',  type=str, default=None,
                        help='Text file listing image paths (one per line)')
    parser.add_argument('--output_dir',  type=str, default='diagnostic_results',
                        help='Where to save masks, scores, and figures')

    # affordance to query
    parser.add_argument('--affordance',  type=str, default='grasp',
                        choices=AFF_LIST,
                        help='Which affordance to query')

    # preprocessing
    parser.add_argument('--crop_size',   type=int, default=448,
                        help='Input resolution (448 matches training)')

    # depth
    parser.add_argument('--use_depth',   action='store_true', default=False,
                        help='Use depth input (requires Depth-Anything or _graydepth.png files)')

    # hardware
    parser.add_argument('--gpu',         type=str, default='0')

    # thresholds (exposed for ablation)
    parser.add_argument('--bg_threshold',       type=float, default=0.8,
                        help='Per-pixel threshold for binary mask (default 0.8 from test.py)')
    parser.add_argument('--global_low_threshold', type=float, default=0.4,
                        help='max_sim below this classifies as identity-like failure')

    args = parser.parse_args()

    # update module-level thresholds from args
    BG_THRESHOLD         = args.bg_threshold
    GLOBAL_LOW_THRESHOLD = args.global_low_threshold

    assert args.image_dir or args.image_list, \
        "Provide either --image_dir or --image_list"

    main(args)

    