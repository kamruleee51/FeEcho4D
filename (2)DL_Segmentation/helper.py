# ======================
# MONAI (Medical Open Network for AI) - medical imaging framework
# ======================
from monai.utils import set_determinism, first              # Utilities: reproducibility, first() helper
from monai.transforms import (                              # Preprocessing & augmentation transforms
    EnsureChannelFirstD,                                    #   Ensure channel-first ordering for dict data
    Compose,                                                #   Compose multiple transforms
    LoadImageD,                                             #   Load image from file (dict interface)
    RandRotateD,                                            #   Random rotation (dict interface)
    RandZoomD,                                              #   Random zoom (dict interface)
    ScaleIntensityRanged                                    #   Intensity scaling within a specified range
)
from monai.data import DataLoader, Dataset, CacheDataset    # Data loading & caching classes
from monai.config import print_config, USE_COMPILED         # Print MONAI/system config, compiled backend flag
from monai.networks.nets import *                           # Built-in network architectures (e.g., UNet, VNet)
from monai.networks.blocks import Warp                      # Warping block (for spatial transforms)
from monai.apps import MedNISTDataset                       # Example dataset loader
from monai.losses import *                                  # Loss functions (Dice, Focal, etc.)
from monai.metrics import *                                 # Evaluation metrics (DiceMetric, etc.)

# ======================
# Albumentations - fast augmentation library
# ======================
import albumentations as A                                  # Augmentation framework
from albumentations.pytorch import ToTensorV2               # Convert Albumentations output to PyTorch tensor

# ======================
# PyTorch core
# ======================
import torch                                                 # Core PyTorch library
import torch.nn as nn                                        # Neural network layers and modules
import torch.nn.functional as F                              # Functional API for layers/activations
import torch.optim as optim                                  # Optimizers (Adam, SGD, etc.)
from torch.autograd import Variable                          # Legacy autograd wrapper (rarely needed now)
from torch.utils.data import Sampler, DataLoader             # Data sampling and loading utilities
from torch.cuda.amp import GradScaler, autocast              # Mixed precision training (automatic casting)
from torchvision import transforms                           # TorchVision transforms (image preprocessing)

# ======================
# Model inspection & visualization
# ======================
from torchinfo import summary                                # Model summary (layers, params, shapes)
from torchviz import make_dot, make_dot_from_trace           # Graph visualization for PyTorch models

# ======================
# Profiling & metrics
# ======================
from fvcore.nn import FlopCountAnalysis                      # FLOPs and parameter count analysis
from piqa import SSIM                                         # Structural Similarity Index (image quality)
import torchmetrics                                           # Common ML metrics (accuracy, IoU, Dice, etc.)

# ======================
# Scientific / numerical stack
# ======================
import numpy as np                                            # Numerical computations
import pandas as pd                                           # Tabular data handling
import matplotlib.pyplot as plt                               # Plotting and visualization
import cv2                                                    # OpenCV for image processing
from skimage.morphology import skeletonize                    # Skeletonization of binary masks
from scipy.ndimage import label, gaussian_filter1d            # Connected components, smoothing
from scipy.spatial.distance import directed_hausdorff         # Directed Hausdorff distance metric

# ======================
# Medical image I/O & visualization
# ======================
import nibabel as nib                                         # NIfTI and other medical image formats
import visdom                                                 # Live experiment dashboard
from tqdm import tqdm                                          # Progress bars for loops
from openpyxl import load_workbook, Workbook                   # Excel file reading/writing

# ======================
# Standard library utilities
# ======================
import random                                                  # Random number generation
import os                                                      # File and path operations
import tempfile                                                # Temporary file utilities
import re                                                      # Regular expressions
from glob import glob                                          # File pattern matching
from typing import List, Tuple, Union                         # Type hints

# ======================
# Local configuration
# ======================
import config                                                  # Project-specific configuration parameters



# -------------------------------------------------------------------
def make_one_hot(labels, device, C=2):
    """
    Converts [B, S, 1, H, W] label tensor to one-hot encoded [B, S, C, H, W].

    Parameters:
        labels : torch.Tensor
            Shape: [B, S, 1, H, W], integer class labels.
        device : torch.device
            Device to place output tensor.
        C : int
            Number of classes.

    Returns:
        one_hot : torch.Tensor
            Shape: [B, S, C, H, W]
    """
    labels = labels.long().to(device)       # Ensure integer class indices and move to target device
    B, S, _, H, W = labels.shape            # Unpack dimensions (ignore the singleton channel)

    one_hot = torch.zeros(B, S, C, H, W, device=device)  # Allocate one-hot tensor on device

    # Scatter needs index in shape [B*S, 1, H, W]
    labels_flat = labels.view(B * S, 1, H, W)            # Merge batch and slice dims for scatter
    one_hot_flat = one_hot.view(B * S, C, H, W)          # Match flattened shape for scatter destination

    one_hot_result = one_hot_flat.scatter_(1, labels_flat, 1)  # Put 1 at class index along channel dim

    # Reshape back to [B, S, C, H, W]
    return one_hot_result.view(B, S, C, H, W)            # Restore original [B,S] structure


# -------------------------------------------------------------------
def show_time_slices(batch, slice_idx: int, num_classes: int, device):
    """
    batch : dict  – a single item from the DataLoader
    slice_idx : int  – which of the 37 spatial slices to display (0-based)
    num_classes : int
    device : torch.device  – the one make_one_hot expects
    """
    imgs  = batch['image']            # [B, 37, 1, H, W] normalized/float images
    masks = batch['mask']             # [B, 37, 1, H, W] integer labels
    boxes = batch['bbox']             # [B, 37, 1, 4]   bounding boxes per slice

    B, _, _, H, W = imgs.shape        # Extract batch size and spatial size for plotting
    n_cols = 2 + num_classes          # Columns: Image | Mask+BBox | one-hot channels per class

    plt.figure(figsize=(5 * n_cols, 5 * B))  # Create a grid proportional to rows/cols

    for t in range(B):                        # Iterate over time frames (batch dimension)
        # ---------- panel 1: input image ---------------------------------
        img = imgs[t, slice_idx, 0].cpu().numpy()           # Select slice (H,W) and move to CPU/NumPy
        img_vis = 255 * (img * 0.5 + 0.5)                   # Simple de-normalization to [0,255] for display
        ax = plt.subplot(B, n_cols, t * n_cols + 1)         # Place subplot at row t, column 1
        ax.imshow(img_vis, cmap='gray')                     # Show grayscale echo image
        ax.set_title(f"T{t}  |  Image")                     # Title with time index
        ax.axis('off')                                      # Hide axes

        # ---------- panel 2: mask + bbox ---------------------------------
        m   = masks[t, slice_idx, 0].cpu().numpy()          # Retrieve mask (H,W)
        bb  = boxes[t, slice_idx, 0].cpu().numpy().astype(int)  # Bounding box [x1,y1,x2,y2]
        x1, y1, x2, y2 = bb                                 # Unpack coordinates
        mask_vis = (m * 50).astype(np.uint8)                # Map labels to small intensities for overlay look
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR) # Convert to 3-channel for rectangle drawing
        mask_vis = cv2.rectangle(mask_vis, (x1, y1), (x2, y2),
                                 (255, 0, 0), 2)            # Draw bbox in blue/red-ish (BGR)
        ax = plt.subplot(B, n_cols, t * n_cols + 2)         # Column 2 for mask+bbox
        ax.imshow(mask_vis)                                 # Show overlay image
        ax.set_title("Mask + BBox")                         # Title
        ax.axis('off')                                      # Hide axes

        # ---------- panels 3…: one-hot channels --------------------------
        # build one-hot for this single slice
        #   shape needed by make_one_hot: [1, 1, H, W]
        m_tensor = masks[t, slice_idx, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add [B,S,1] dims as [1,1,1,H,W]
        one_hot  = make_one_hot(m_tensor, device, num_classes)[0, 0]  # Compute one-hot, then select [C,H,W]

        for c in range(num_classes):                       # Iterate over classes
            ax = plt.subplot(B, n_cols, t * n_cols + 3 + c)  # Place subplot for class c
            ax.imshow(one_hot[c].cpu().numpy(), cmap='gray') # Visualize class mask
            ax.set_title(f"Class {c}")                       # Label class index
            ax.axis('off')                                   # Hide axes

    plt.tight_layout()                                      # Improve layout
    plt.show()                                              # Render figure

# --------------------------------------------------------------------------
def show_slices(batch,
                volume_idx: int,                 # which time-frame (T)
                slice_indices,                   # iterable of slice IDs
                num_classes: int,
                device):
    """
    Visualise multiple spatial slices of ONE time-frame from a batch.

    Parameters
    ----------
    batch : dict  (keys = 'image', 'mask', 'bbox', 'name')
        A single output from the DataLoader.
    volume_idx : int
        Index in the batch -> which time-frame to fix.
    slice_indices : list[int] or range
        Which of the 37 slices to draw (0-based).
    num_classes : int
        Number of semantic classes in make_one_hot().
    device : torch.device
        Device expected by make_one_hot().
    """
    imgs  = batch['image']           # [B, 37, 1, H, W] images
    masks = batch['mask']            # [B, 37, 1, H, W] labels
    boxes = batch['bbox']            # [B, 37, 1, 4]   bboxes

    _, _, _, H, W = imgs.shape       # Spatial size for plotting scale
    n_rows = len(slice_indices)      # One row per slice index
    n_cols = 2 + num_classes         # Image | Mask+BBox | one-hot channels

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))  # Configure figure size

    for r, s_idx in enumerate(slice_indices):     # Iterate over selected slices
        # -------- panel 1 : image -----------------------------------------
        img = imgs[volume_idx, s_idx, 0].cpu().numpy()      # Extract image slice
        img_vis = 255 * (img * 0.5 + 0.5)                   # De-normalize to display range
        ax = plt.subplot(n_rows, n_cols, r * n_cols + 1)    # Place subplot (row r, col 1)
        ax.imshow(img_vis, cmap='gray')                     # Show image
        ax.set_title(f"Slice {s_idx:02d} | Image")          # Title with slice id
        ax.axis('off')                                      # Hide axes

        # -------- panel 2 : mask + bbox -----------------------------------
        m   = masks[volume_idx, s_idx, 0].cpu().numpy()     # Mask slice
        bb  = boxes[volume_idx, s_idx, 0].cpu().numpy().astype(int)  # BBox coords
        x1, y1, x2, y2 = bb                                 # Unpack bbox
        mask_vis = (m * 50).astype(np.uint8)                # Map labels to small intensities
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)       # To color image
        mask_vis = cv2.rectangle(mask_vis, (x1, y1), (x2, y2),
                                 (255, 0, 0), 2)            # Draw bbox
        ax = plt.subplot(n_rows, n_cols, r * n_cols + 2)    # Column 2
        ax.imshow(mask_vis)                                 # Show overlay
        ax.set_title("Mask + BBox")                         # Title
        ax.axis('off')                                      # Hide axes

        # -------- panels 3… : one-hot channels ----------------------------
        m_tensor = masks[volume_idx, s_idx, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,H,W]
        one_hot = make_one_hot(m_tensor, device, num_classes)[0, 0]  # [C,H,W]

        for c in range(num_classes):                        # Loop classes
            ax = plt.subplot(n_rows, n_cols, r * n_cols + 3 + c)  # Place subplot
            ax.imshow(one_hot[c].cpu().numpy(), cmap='gray')      # Show class map
            ax.set_title(f"Class {c}")                            # Title
            ax.axis('off')                                        # Hide axes

    plt.tight_layout()                                        # Adjust spacing
    plt.show()                                                # Display figure



def read_img(path):
    img_ = cv2.imread(path, 0)  # Read grayscale image from disk (uint8)
    img_ = cv2.resize(img_, (config.img_size, config.img_size), interpolation=cv2.INTER_CUBIC)  # Resize with bicubic
    return img_                 # Return resized image (uint8)


def read_mask(path):
    mask_ = cv2.imread(path, 0)  # Read mask as grayscale (label values encoded in intensity)
    mask_ = cv2.resize(mask_, (config.img_size, config.img_size), interpolation=cv2.INTER_NEAREST)  # Nearest preserves labels
    mask_[mask_ == 0] = 0        # Map background to 0 (explicit, keeps as is)
    mask_[mask_ == 127] = 2      # Map mid-gray to label 2 (e.g., myocardium/cavity convention)
    mask_[mask_ == 255] = 1      # Map white to label 1
    return mask_                 # Return relabeled mask (uint8)
    

def summary(name, imgs, msks, expected_slices=37):
    assert len(imgs) == len(msks), f"{name}: image/mask count mismatch"               # Ensure paired lists
    for k, vol in enumerate(imgs):                                                    # Iterate volumes
        assert len(vol) == expected_slices, f"{name}: volume {k} has {len(vol)} slices"  # Check slice count
    print(f"{name}: {len(imgs)} complete volumes ✔️")                                 # Report summary

# ========== (1) filename utilities ==========================================

def get_volume_prefix_suffix(filename):
    """
    Splits "...sliceXXX..." into  prefix + suffix.  Works with:
      12OCT2021P2_4_slice001_time_001.png
      Patient001_slice001time001.png
    """
    base = os.path.basename(filename)                     # Get basename without dirs
    m = re.search(r'(slice\d{3})', base)                  # Find 'sliceNNN' token
    if m is None:                                         # If not found, error out
        raise ValueError(f"'sliceXXX' not found in {filename}")
    start, end = m.span()                                 # Span of the match
    return base[:start], base[end:]                       # Return parts around the token

# patient/time extractor for the new sampler
_name_re = re.compile(r'^(?P<pid>[^_]+)_slice\d{3}.*?time(?P<time>\d+)\.png$')  # Regex: PID and time index from name

def get_pid_time(path: str):
    m = _name_re.match(os.path.basename(path))            # Match pattern against filename
    if m is None:                                         # Validate
        raise ValueError(f"Bad filename pattern: {path}")
    pid   = m.group('pid')                                # Extract patient ID
    t_idx = int(m.group('time')) - 1                      # Convert time to 0-based index
    return pid, t_idx                                     # Return (patient, time)

def collect_complete_volumes(
    data_dir: str,
    split: Union[str, List[str]] = "train",
    num_slices: int = 37,
    slice_token: str = "slice",
    verbose: bool = True
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Return only volumes that contain *all* `num_slices` image–mask pairs
    for the requested dataset split(s).

    Parameters
    ----------
    data_dir : str
        Root directory that contains sub-folders `train/`, `val/`, `test/`.
    split : str | List[str], default = "train"
        Which split(s) to scan.  Examples: "val"  or  ["val", "test"].
    num_slices : int, default = 37
    slice_token : str, default = "slice"
    verbose : bool, default = True

    Returns
    -------
    chunked_images, chunked_masks : List[List[str]], List[List[str]]
        Parallel lists of complete volumes.
    """
    # ----------------------------------------------------------- #
    if isinstance(split, str):
        splits = [split]                                           # Normalize to list
    else:
        splits = split                                             # Already a list

    chunked_images, chunked_masks = [], []                         # Accumulate complete volumes

    for sp in splits:                                              # Iterate requested splits
        img_seed_paths = sorted(
            glob(os.path.join(data_dir, f"{sp}/image/*/*{slice_token}001*.png"))
        )                                                          # Find seed slice001 image files
        msk_seed_paths = sorted(
            glob(os.path.join(data_dir, f"{sp}/mask/*/*{slice_token}001*.png"))
        )                                                          # Find corresponding mask seeds

        for img001, msk001 in zip(img_seed_paths, msk_seed_paths): # Iterate paired seeds
            try:
                prefix, suffix = get_volume_prefix_suffix(os.path.basename(img001))  # Split around 'sliceNNN'
                img_dir, msk_dir = os.path.dirname(img001), os.path.dirname(msk001) # Parent dirs

                img_chunk, msk_chunk = [], []                      # Temporary holders for this volume
                for i in range(1, num_slices + 1):                 # Loop over all slice indices
                    tag   = f"{slice_token}{str(i).zfill(3)}"      # Build slice token e.g., slice001
                    fname = prefix + tag + suffix                  # Reconstruct filename
                    ipath = os.path.join(img_dir, fname)           # Image path for slice i
                    mpath = os.path.join(msk_dir, fname)           # Mask path for slice i
                    if os.path.exists(ipath) and os.path.exists(mpath):  # Both must exist
                        img_chunk.append(ipath)                     # Keep image slice
                        msk_chunk.append(mpath)                     # Keep mask slice
                    else:
                        if verbose:
                            print(f"[{sp}] Missing: {ipath} or {mpath}")   # Report missing slice
                        break                                      # Incomplete → stop collecting this volume

                if len(img_chunk) == num_slices:                   # If complete set found
                    chunked_images.append(img_chunk)               # Save volume image paths
                    chunked_masks.append(msk_chunk)                # Save volume mask paths

            except ValueError as e:
                if verbose: print(e)                               # Print filename parsing errors

    # summary
    if verbose:
        n_img, n_msk = len(chunked_images), len(chunked_masks)     # Counts for sanity check
        print(f"Total volumes ({'+'.join(splits)}): {n_img}")      # Report how many volumes found
        if n_img != n_msk:
            print("⚠️  Image / mask count mismatch!")              # Warn if mismatched

    return chunked_images, chunked_masks                           # Return parallel lists


def mask_to_scribble(
    mask: np.ndarray,
    *,
    target_label: int = 1,
    tail_frac: float = 0.1,
    big_amp=(15, 25),
    small_amp=(6, 12),
    big_knot_num=(6, 12),
    big_smooth=(15, 25),
    small_smooth=3,
    line_width=(2, 4)
) -> np.ndarray:
    """
    Generate a distorted centerline scribble from a given mask label region.

    Parameters
    ----------
    mask         : 2D uint8 input mask image.
    target_label : int       – target label to generate scribble for (e.g., MYO).
    tail_frac    : float     – fraction of skeleton to extend at each end.
    big_amp      : int/tuple – amplitude of large-scale distortions.
    small_amp    : int/tuple – amplitude of small-scale noise perturbations.
    big_knot_num : int/tuple – number of anchor points for large-scale distortions.
    big_smooth   : int/tuple – Gaussian smoothing factor for large distortions.
    small_smooth : int       – Gaussian smoothing for small noise.
    line_width   : int/tuple – output scribble thickness in pixels.

    Returns
    -------
    scribble     : 2D uint8 image with values {0, 1}, shape same as input mask.
    """

    def get_param(p):
        return random.randint(*p) if isinstance(p, tuple) else p   # Resolve fixed value or random in range

    def keep_largest_region(mask):
        labeled, num = label(mask)                                 # Label connected components
        if num == 0:
            return mask                                            # No components → return as is
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1     # Find largest component id
        return labeled == largest                                  # Keep only largest region

    def extrapolate_tail(skel: np.ndarray, tail_frac=0.1):
        h, w = skel.shape                                          # Image size
        pts = np.argwhere(skel)                                     # Get skeleton pixel coords
        if pts.size == 0:
            return skel.copy()                                     # Empty skeleton → nothing to do
        xs = pts[:, 1]                                             # X coordinates
        x_mid = np.median(xs)                                      # Median X for splitting left/right
        sides = [pts[xs <= x_mid], pts[xs > x_mid]]                # Two sides
        out = skel.copy()                                          # Working copy
        for side in sides:                                         # Process each side
            if len(side) < 6:
                continue                                           # Too short to extrapolate
            y_min, y_max = side[:, 0].min(), side[:, 0].max()      # Vertical extent
            span = y_max - y_min                                   # Length along y
            if span < 5:
                continue                                           # Too short span
            y_cut = y_max - 0.1 * span                             # Trim tip region to stabilize fit
            out[side[side[:, 0] >= y_cut][:, 0], side[side[:, 0] >= y_cut][:, 1]] = False  # Erase tip
            fit_mask = (side[:, 0] >= y_max - 0.3 * span) & (side[:, 0] <= y_max - 0.1 * span)  # Fit window
            fit_pts = side[fit_mask]                               # Points used to fit a line
            if len(fit_pts) < 3:
                continue                                           # Need ≥ 3 points
            y_fit = fit_pts[:, 0].astype(float)                    # y inputs
            x_fit = fit_pts[:, 1].astype(float)                    # x observations
            A = np.vstack([y_fit, np.ones_like(y_fit)]).T          # Design matrix for linear fit
            a, b = np.linalg.lstsq(A, x_fit, rcond=None)[0]        # Least-squares slope/intercept
            y_new = np.arange(int(round(y_max - 0.2 * span)), y_max + 1)  # New y to extend
            x_new = np.clip(np.rint(a * y_new + b).astype(int), 0, w - 1) # Predicted x, clamped
            out[y_new, x_new] = True                               # Draw extension
        return skeletonize(out)                                     # Re-skeletonize after edits

    def extract_ordered_path(skel):
        h, w = skel.shape                                          # Image size
        pts = np.argwhere(skel)                                     # Skeleton points
        if pts.size == 0:
            return []                                              # No path
        neigh8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]  # 8-connectivity
        sk_mask = skel.astype(bool)                                # Boolean mask for neighbors
        visited, path = set(), []                                  # DFS state
        degrees = {tuple(p): sum(sk_mask[p[0]+dy, p[1]+dx]
                                 if 0<=p[0]+dy<h and 0<=p[1]+dx<w else 0
                                 for dy,dx in neigh8)
                   for p in pts}                                   # Node degree map
        start = next((p for p,d in degrees.items() if d==1), pts[0])  # Start at an endpoint if possible
        stack = [tuple(start)]                                     # DFS stack
        while stack:
            y, x = stack.pop()                                     # Visit node
            if (y, x) in visited:
                continue                                           # Skip revisits
            visited.add((y, x))                                    # Mark visited
            path.append((y, x))                                    # Append to path
            for dy, dx in neigh8:                                  # Explore neighbors
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and sk_mask[ny, nx] and (ny, nx) not in visited:
                    stack.append((ny, nx))                         # Push first unvisited neighbor
                    break
        return path                                                # Return ordered path coordinates

    def distort_skeleton_line(skel_img: np.ndarray) -> np.ndarray:
        h, w = skel_img.shape                                      # Image size
        path = extract_ordered_path(skel_img > 0)                  # Ordered path along skeleton
        if len(path) < 2:
            # Minimal fallback scribble at least draws one pixel
            canvas = np.zeros((h, w), np.uint8)                    # Blank canvas
            yx = np.argwhere(skel_img > 0)                         # Any positive pixel?
            if yx.size > 0:
                y, x = yx[0]                                       # Take first pixel
                cv2.circle(canvas, (x, y), radius=1, color=255, thickness=1)  # Draw dot
            return canvas                                          # Return minimal scribble

        path = np.array(path, dtype=np.float32)                    # Convert to float path
        N = len(path)                                              # Number of points
        envelope = np.zeros(N, np.float32)                         # Large-scale offset envelope

        # Ensure knot_num ≤ N to avoid ValueError
        requested_knot_num = get_param(big_knot_num)               # Random knots requested
        knot_num = min(N, max(1, requested_knot_num))              # Safe clamp between [1, N]
        knot_idx = np.random.choice(N, knot_num, replace=False)    # Sample knot indices

        envelope[knot_idx] = np.random.uniform(
            -get_param(big_amp), get_param(big_amp), size=knot_num
        )                                                          # Assign random offsets at knots
        envelope = gaussian_filter1d(envelope, sigma=get_param(big_smooth))  # Smooth envelope
        envelope /= (np.max(np.abs(envelope)) + 1e-6)              # Normalize to [-1,1]
        envelope *= get_param(big_amp)                             # Rescale to amplitude

        micro = np.random.uniform(-1, 1, N)                        # Small-scale noise
        micro = gaussian_filter1d(micro, sigma=small_smooth)       # Smooth noise
        micro /= (np.max(np.abs(micro)) + 1e-6)                    # Normalize to [-1,1]
        micro *= get_param(small_amp)                              # Scale to small_amp

        total_offset = envelope + micro                            # Combine large + small offsets

        diff = np.vstack([path[1] - path[0], path[2:] - path[:-2], path[-1] - path[-2]])  # Approx tangent
        tang = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6)                # Unit tangent
        normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)                               # Perp normal

        pert_xy = np.stack([path[:, 1], path[:, 0]], axis=1) + total_offset[:, None] * normal  # Offset path

        canvas = np.zeros((h, w), np.uint8)                          # Output canvas
        cv2.polylines(canvas,
                    [pert_xy.astype(np.int32)],
                    isClosed=False,
                    color=255,
                    thickness=get_param(line_width),
                    lineType=cv2.LINE_AA)                            # Draw polyline as scribble
        return canvas                                                # Return rendered scribble


    # --- Main logic ---
    region = (mask == target_label).astype(np.uint8)              # Binary region for target label
    region = keep_largest_region(region)                          # Keep only largest connected component
    skel = skeletonize(region)                                    # Thin to 1-pixel skeleton
    skel = extrapolate_tail(skel, tail_frac=tail_frac)            # Extend both ends slightly
    skel_u8 = (skel.astype(np.uint8)) * 255                       # Convert to 0/255 for drawing
    scribble = distort_skeleton_line(skel_u8)                     # Generate perturbed path scribble
    return (scribble > 0).astype(np.uint8)                        # Return binary 0/1 image


def myo_scribble_human(
        mask: np.ndarray,
        target_label: int = 1,
        thickness: int = 2,
        jitter: int = 2,
        blur_sigma: float = 0,
        binary_output: bool = False,
    ) -> np.ndarray:
    """
    Generate a thick, slightly irregular centre-line scribble.

    Parameters
    ----------
    mask          : 2-D uint8 label map.
    target_label  : int   – label value that encodes MYO.
    thickness     : int   – approximate stroke width in pixels.
    jitter        : int   – max ±pixel offset per skeleton point.
    blur_sigma    : float – Gaussian σ for soft edge (0 = crisp).
    binary_output : bool  – if True, return 0/1; else 0/255.

    Returns
    -------
    scribble      : uint8 image of the stroke.
    """

    # ── 1. keep only MYO │ (0/1 image) ───────────────────────────────────────
    myo_bin = np.where(mask == target_label, 1, 0).astype(np.uint8)                # Binary mask for target label

    # ── 2. centre-line skeleton (opencv-contrib or fallback) ────────────────
    try:
        skel = cv2.ximgproc.thinning(myo_bin * 255,
                                     cv2.ximgproc.THINNING_ZHANGSUEN)             # Use ximgproc thinning if available
        skel = (skel > 0).astype(np.uint8)                                         # Convert to 0/1
    except AttributeError:                      # if ximgproc missing
        skel = skeletonize(myo_bin).astype(np.uint8)                               # Fallback to skimage skeletonize

    # ── 3. paint jittered circles along skeleton ────────────────────────────
    h, w = skel.shape                                                               # Image size
    scribble = np.zeros((h, w), np.uint8)                                           # Output canvas

    ys, xs = np.nonzero(skel)                                                       # Skeleton pixel coordinates
    radius = max(1, thickness // 2)                                                 # Circle radius from thickness

    for y, x in zip(ys, xs):                                                        # Draw small discs along skeleton
        jx = np.clip(x + random.randint(-jitter, jitter), 0, w - 1)                 # Jitter x within bounds
        jy = np.clip(y + random.randint(-jitter, jitter), 0, h - 1)                 # Jitter y within bounds
        cv2.circle(scribble, (jx, jy), radius, 255, -1)                             # Paint filled circle

    # ── 4. close tiny gaps then (optionally) soften edges ───────────────────
    k = max(3, radius) | 1                          # odd kernel size             # Ensure kernel size is odd ≥3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))                  # Elliptical structuring element
    scribble = cv2.morphologyEx(scribble, cv2.MORPH_CLOSE, kernel)                  # Close small gaps

    if blur_sigma and blur_sigma > 0:                                               # Optional softening
        k_blur = int(blur_sigma * 6) | 1            # odd kernel size              # Gaussian kernel size (odd)
        scribble = cv2.GaussianBlur(scribble, (k_blur, k_blur), blur_sigma)         # Apply blur

    # ── 5. 0/1 or 0/255? ────────────────────────────────────────────────────
    if binary_output:
        return (scribble > 0).astype(np.uint8)      # 0 / 1                       # Return binary
    else:
        return (scribble > 0).astype(np.uint8) * 255                               # Return 0/255 mask

 
def generate_scribble_segments(
    mask: np.ndarray,
    target_label: int,
    segment_num: Tuple[int, int],
    wave_num: Tuple[int, int],
    amp_frac: Tuple[float, float],
    max_hide: int,
    resample_step: int,
    min_amp_px: float,
    smooth_sigma: float,
    line_width: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """
    Generate a distorted skeleton scribble from a binary mask using segment-wise large wave perturbations.
 
    Args:
        mask (np.ndarray): 2D binary mask input.
        target_label (int): Target label value in the mask to operate on.
        segment_num (Tuple[int, int]): Range of number of segments to split the skeleton path.
        wave_num (Tuple[int, int]): Range of wave counts per segment.
        amp_frac (Tuple[float, float]): Range of amplitude factors for offset (multiplied with distance transform).
        max_hide (int): Max number of segments randomly dropped from drawing.
        resample_step (int): Step size for resampling the skeleton path.
        min_amp_px (float): Minimum effective amplitude threshold in pixels.
        smooth_sigma (float): Final Gaussian smoothing for the perturbed path.
        line_width (int or Tuple[int, int]): Fixed or random line thickness in pixels.
 
    Returns:
        np.ndarray: 2D scribble mask (uint8) with shape same as input mask, values {0, 1}.
    """
    if isinstance(line_width, tuple):
        line_px = random.randint(*line_width)                                          # Randomize line width in range
    else:
        line_px = max(1, int(line_width))                                              # Clamp to ≥1
 
    region = (mask == target_label).astype(np.uint8)                                   # Binary region of interest
    num, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)                # Connected components
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])                           # Select largest component id
        region = (labels == largest).astype(np.uint8)                                  # Keep only largest component
    if region.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)                                     # Empty region → empty scribble
 
    skel = skeletonize(region).astype(np.uint8)                                        # Skeletonize ROI
    pts = np.argwhere(skel)                                                            # Skeleton points
    if pts.size < 2:
        return np.zeros_like(mask, dtype=np.uint8)                                     # Too small to draw
 
    h, w = skel.shape                                                                  # Dimensions
    neigh8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]                   # 8-neighborhood
    visited, path = set(), []                                                          # DFS state
    deg = {tuple(p): sum(skel[p[0]+dy, p[1]+dx] if 0<=p[0]+dy<h and 0<=p[1]+dx<w else 0
                         for dy,dx in neigh8) for p in pts}                            # Degree per pixel
    start = next((p for p,d in deg.items() if d==1), pts[0])                           # Start at endpoint if possible
    stack = [tuple(start)]                                                             # DFS stack
    while stack:
        y, x = stack.pop()                                                             # Pop next node
        if (y,x) in visited: continue                                                  # Skip if seen
        visited.add((y,x)); path.append([y,x])                                         # Mark and record
        for dy, dx in neigh8:
            ny, nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in visited:
                stack.append((ny,nx)); break                                           # Push first neighbor then break
    path = np.asarray(path, np.float32)                                                # Convert to float path
 
    arc = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))                     # Cumulative arc-length
    arc = np.insert(arc, 0, 0)                                                         # Start at 0
    s_new = np.arange(0, arc[-1]+1e-6, resample_step)                                  # Uniform samples along path
    y_new = np.interp(s_new, arc, path[:,0])                                           # Interpolate y along arc
    x_new = np.interp(s_new, arc, path[:,1])                                           # Interpolate x along arc
    path  = np.stack([y_new, x_new], 1)                                                # Resampled path (N,2)
    N     = len(path)                                                                   # Number of points
 
    seg_n = random.randint(*segment_num)                                               # Random number of segments
    idx   = np.linspace(0, N, seg_n+1, dtype=int)                                      # Segment boundaries
    segs  = [path[idx[i]:idx[i+1]] for i in range(seg_n)]                              # Split path into segments
 
    hide_cnt = random.randint(0, min(max_hide, seg_n-1)) if max_hide > 0 else 0        # Randomly drop some segments
    hide_set = set(random.sample(range(seg_n), hide_cnt)) if hide_cnt else set()       # Indices to hide
 
    scribble = np.zeros_like(mask, dtype=np.uint8)                                     # Output canvas
    dist_map = cv2.distanceTransform(region, cv2.DIST_L2, 5)                           # Distance to boundary
 
    for i, seg in enumerate(segs):
        if i in hide_set or len(seg) < 2:
            continue                                                                   # Skip hidden or tiny segments
        seg_len = np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))              # Arc-length within segment
        seg_len = np.insert(seg_len, 0, 0)
        L = seg_len[-1] if seg_len[-1] > 0 else 1.0                                    # Avoid divide-by-zero
 
        k   = random.randint(*wave_num)                                                # Wave count
        phi = random.uniform(0, 2*np.pi)                                               # Phase
        amp = random.uniform(*amp_frac)                                                # Amplitude fraction
        dist_seg = dist_map[seg[:,0].astype(int), seg[:,1].astype(int)]                # Distances along path
        offset = amp * dist_seg * np.sin(2*np.pi*k*seg_len/L + phi)                    # Normal offset magnitude
        offset[np.abs(offset) < min_amp_px] = 0                                        # Suppress tiny offsets
 
        diff = np.vstack([seg[1]-seg[0], seg[2:]-seg[:-2], seg[-1]-seg[-2]])           # Approx tangent per point
        tang = diff / (np.linalg.norm(diff, axis=1, keepdims=True)+1e-6)               # Unit tangent
        norm = np.stack([tang[:,1], -tang[:,0]], axis=1)                                # Normal vectors
        pert = seg + offset[:,None]*norm                                               # Apply offset
 
        pert[:,0] = gaussian_filter1d(pert[:,0], smooth_sigma, mode='nearest')         # Smooth y
        pert[:,1] = gaussian_filter1d(pert[:,1], smooth_sigma, mode='nearest')         # Smooth x
 
        cv2.polylines(
            scribble,
            [pert[:, ::-1].astype(np.int32)],                                          # Note: cv2 expects (x,y)
            isClosed=False,
            color=1,
            thickness=line_px,
            lineType=cv2.LINE_AA
        )                                                                              # Draw perturbed segment
    return scribble.astype(np.uint8)                                                   # Return binary (0/1) canvas

class ReadChunkDatasetScribble(Dataset):
    """
    image_chunks : list[list[str]]
        Outer list = one 3-D volume; inner list = slice paths
    mask_chunks  : same structure as image_chunks

    transform    : Albumentations.Compose with
        additional_targets = {'scribble': 'mask'}
    """
    def __init__(self, image_chunks, mask_chunks, transform=None):
        self.image_chunks = image_chunks                      # Store list of image path lists (per volume)
        self.mask_chunks  = mask_chunks                       # Store list of mask path lists (per volume)
        self.transform    = transform                         # Optional Albumentations transform

    def __len__(self):
        return len(self.image_chunks)                         # Number of volumes

    def __getitem__(self, idx):
        img_paths = self.image_chunks[idx]                    # Paths for volume slices (images)
        msk_paths = self.mask_chunks[idx]                     # Paths for volume slices (masks)

        imgs, msks, scribs, names = [], [], [], []            # Accumulators for this volume

        for ipath, mpath in zip(img_paths, msk_paths):        # Iterate slice pairs
            img = read_img(ipath)          # H×W uint8  (0-255)               # Load image (resized)
            msk = read_mask(mpath)         # H×W uint8  (label map)           # Load mask (resized/relabelled)
            scribble = generate_scribble_segments(
                                                msk,
                                                target_label = 1,
                                                segment_num=(1, 3),
                                                wave_num=(1, 2),
                                                amp_frac=(0.1, 1.8),
                                                max_hide=1,
                                                resample_step=2,
                                                min_amp_px=1.0,
                                                smooth_sigma=3.0,
                                                line_width=(3, 10)
                                            )                                     # Synthesize a scribble for this slice
            # Albumentations transform (if any)
            if self.transform is not None:
                res = self.transform(
                    image=img.astype(np.uint8),
                    mask=msk.astype(np.uint8),
                    scribble=scribble.astype(np.uint8)
                )                                                                # Apply identical spatial transforms
                img_t  = res['image'].float()                                    # Tensor image (C,H,W) float
                msk_t  = res['mask'].float().unsqueeze(0)                        # Add channel dim → (1,H,W)
                scr_t  = res['scribble'].float().unsqueeze(0)                    # Add channel dim → (1,H,W)
            else:
                img_t = torch.tensor(img,      dtype=torch.float32)[None]        # (1,H,W) if no transform
                msk_t = torch.tensor(msk,      dtype=torch.float32)[None]        # (1,H,W)
                scr_t = torch.tensor(scribble, dtype=torch.float32)[None]        # (1,H,W)

            imgs.append(img_t)                                                   # Accumulate tensors
            msks.append(msk_t)
            scribs.append(scr_t)
            names.append(ipath)                                                  # Keep original path (name)

        return dict(
            image    = torch.stack(imgs),    # [D, 1, H, W]                   # Stack along slice dimension
            mask     = torch.stack(msks),    # [D, 1, H, W]
            scribble = torch.stack(scribs),  # [D, 1, H, W]
            name     = names                 # List[str] of slice paths
        )



class ReadChunkDataset(Dataset):
    def __init__(self, image_chunks, mask_chunks, transform=None):
        self.image_chunks = image_chunks                      # Volume-wise list of slice path lists (images)
        self.mask_chunks  = mask_chunks                       # Volume-wise list of slice path lists (masks)
        self.transform    = transform                         # Optional Albumentations (with bbox support)

    def __len__(self):
        return len(self.image_chunks)                         # Number of volumes

    def __getitem__(self, idx):
        img_paths  = self.image_chunks[idx]                   # Slice paths for this volume (images)
        msk_paths  = self.mask_chunks[idx]                    # Slice paths for this volume (masks)

        imgs, msks, bboxes, names = [], [], [], []            # Accumulators per volume
        for ipath, mpath in zip(img_paths, msk_paths):        # Iterate slice pairs
            img  = read_img(ipath)                            # Load image (uint8)
            msk  = read_mask(mpath)                           # Load mask  (uint8 labels)
            mask1 = (msk != 0).astype(np.uint8)               # Binary foreground mask
            box_list = get_bounding_boxes_from_mask(mask1)    # Extract list of bboxes [x1,y1,x2,y2]

            if self.transform and len(box_list) > 0:
                labels = [1] * len(box_list)                  # Dummy labels for Albumentations bbox API
                res = self.transform(image=img.astype(np.uint8),
                                     mask=msk.astype(np.uint8),
                                     bboxes=box_list,
                                     labels=labels)           # Apply transforms consistently to img/mask/bboxes
                img_t  = res['image'].float()                 # Transformed image tensor (C,H,W)
                msk_t  = res['mask'].float().unsqueeze(0)     # (1,H,W) mask tensor
                box_t  = torch.tensor(res['bboxes'], dtype=torch.float32)  # Transformed bboxes
            else:
                img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W) image tensor
                msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)  # (1,H,W) mask tensor
                box_t = torch.tensor(box_list, dtype=torch.float32)          # Original bboxes (possibly empty)

            if box_t.ndim == 1: box_t = box_t.unsqueeze(0)    # Ensure shape [N,4] even if single box
            imgs.append(img_t)                                 # Accumulate tensors
            msks.append(msk_t)
            bboxes.append(box_t)
            names.append(ipath)                                # Keep path as name

        return dict(image = torch.stack(imgs),     # [37,1,H,W]    # Stack along slice dim
                    mask  = torch.stack(msks),     # [37,1,H,W]
                    bbox  = torch.stack(bboxes),   # [37,1,4]      # One bbox per slice (or empty)
                    name  = names)                 # List of paths



def get_bounding_boxes_from_mask(mask, perturbation=True, max_perturb=20, min_area=50):
    mask = (mask * 255).astype(np.uint8)                                       # Convert 0/1 to 0/255 uint8
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find outer contours

    H, W = mask.shape                                                           # Image size
    bboxes = []                                                                 # Collected boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)                                             # Area of contour
        if area < min_area:
            continue  # Skip tiny blobs/noise                                    # Filter small regions

        x, y, w, h = cv2.boundingRect(cnt)                                      # Axis-aligned bbox

        if perturbation:
            x_min = max(0, x - np.random.randint(0, max_perturb))               # Randomly expand left/top
            y_min = max(0, y - np.random.randint(0, max_perturb))
            x_max = min(W, x + w + np.random.randint(0, max_perturb))           # Randomly expand right/bottom
            y_max = min(H, y + h + np.random.randint(0, max_perturb))
        else:
            x_min, y_min, x_max, y_max = x, y, x + w, y + h                     # Exact bbox

        bboxes.append([x_min, y_min, x_max, y_max])                              # Append [x1,y1,x2,y2]
    return bboxes                                                                # Return list of boxes





# ========== (4) custom BatchSampler (stride-aware) ===========================
class NeighborTimeBatchSampler(Sampler):
    """
    Yield batches of `window_size` consecutive time-frames from ONE patient,
    stepping forward by `stride`.

      • stride = 1              → max overlap
      • stride = window_size    → non-overlapping blocks
      • 1 < stride < window_size → partial overlap
    If the remaining frames are fewer than `window_size`,
    they are yielded as a smaller tail batch when drop_last=False.
    """
    def __init__(self, p2idxs, *,          # force keyword args after p2idxs
                 window_size: int = 5,
                 stride: int = 1,
                 shuffle_patients: bool = True,
                 drop_last: bool = False):
        assert window_size >= 1                                                # Validate window size
        assert stride      >= 1                                                # Validate stride
        self.p2idxs           = p2idxs                                         # Dict[pid] -> List[(time, index)]
        self.window_size      = window_size                                    # Sliding window length
        self.stride           = stride                                         # Step size between windows
        self.shuffle_patients = shuffle_patients                               # Shuffle patient order each epoch
        self.drop_last        = drop_last                                      # Drop incomplete tails if True
        self.pids             = list(self.p2idxs.keys())                       # Cache patient ids

    # ------------------------------------------------------------------------
    def __iter__(self):
        if self.shuffle_patients:
            random.shuffle(self.pids)                                          # Optional patient-level shuffling

        for pid in self.pids:
            idx_sorted = [idx for _, idx in self.p2idxs[pid]]   # by time      # Extract indices ordered by time
            n = len(idx_sorted)                                                # Number of frames for patient

            # main sliding windows
            for start in range(0, n - self.window_size + 1, self.stride):
                yield idx_sorted[start : start + self.window_size]             # Yield consecutive window

            # tail (fewer than window_size left)
            remainder = n % self.stride                                        # Frames not aligned to stride
            tail_len  = n - (n // self.stride) * self.stride                   # Compute tail length
            if remainder and not self.drop_last:
                tail = idx_sorted[-tail_len:]                                  # Take last remainder frames
                yield tail                                                     # Yield tail batch

    # ------------------------------------------------------------------------
    def __len__(self):
        total = 0                                                              # Total number of yielded batches
        for pid in self.pids:
            n = len(self.p2idxs[pid])                                          # Frames in this patient
            if n < self.window_size:
                total += 0 if self.drop_last else 1                            # Either drop or count single tail
            else:
                total += ((n - self.window_size) // self.stride) + 1           # Full windows count
                if (n - self.window_size) % self.stride and not self.drop_last:
                    total += 1                                                 # Plus one tail if applicable
        return total                                                           # Return number of batches



# ---------------------------------------------
def show_time_slices_scribble(batch, slice_idx: int, num_classes: int, device):
    """
    Visualise all time-frames (rows) for a single spatial slice.

    Parameters:
        batch        : dict with keys 'image', 'mask', 'scribble', 'name'
        slice_idx    : int, index of the radial slice to display (0 to S-1)
        num_classes  : int, number of classes in segmentation
        device       : torch.device, used for one-hot encoding
    """
    imgs   = batch['image']       # [B, S, 1, H, W] image tensor
    masks  = batch['mask']        # [B, S, 1, H, W] label tensor
    scrs   = batch['scribble']    # [B, S, 1, H, W] scribble tensor

    B, _, _, H, W = imgs.shape    # Extract dimensions for plotting
    n_cols = 2 + num_classes      # Columns: Image | Image+Scribble | per-class one-hot

    plt.figure(figsize=(4 * n_cols, 4 * B))  # Create overall figure

    for t in range(B):  # Loop over time frames
        # --- Panel 1: Echo image ---
        img = imgs[t, slice_idx, 0].cpu().numpy()                              # Get slice image
        img_vis = np.clip(255 * (img * 0.5 + 0.5), 0, 255).astype(np.uint8)    # De-normalize + clamp

        ax = plt.subplot(B, n_cols, t * n_cols + 1)                            # Place subplot (row t, col 1)
        ax.imshow(img_vis, cmap='gray')                                        # Show grayscale
        ax.set_title(f"T{t} | Image")                                          # Title with time index
        ax.axis('off')                                                         # Hide axes

        # --- Panel 2: Image + Scribble ---
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)                    # To 3-channel for overlay

        s = scrs[t, slice_idx, 0].cpu().numpy()                                # Scribble mask
        s_bin = (s > 0).astype(np.uint8)                                       # Binarize scribble
        s_vis = cv2.dilate(s_bin, np.ones((3, 3), np.uint8), iterations=1)     # Thicken for visibility
        img_rgb[s_vis > 0] = (0, 0, 255)                                       # Paint red where scribble present

        ax = plt.subplot(B, n_cols, t * n_cols + 2)                            # Place subplot (row t, col 2)
        ax.imshow(img_rgb)                                                     # Show overlay
        ax.set_title("Image + Scribble")                                       # Title
        ax.axis('off')                                                         # Hide axes

        # --- Panels 3…: One-hot channels ---
        m_slice = masks[t, slice_idx, 0][None, None, None]  # [1,1,1,H,W]      # Build singleton batch for one-hot
        one_hot = make_one_hot(m_slice.to(device), device, num_classes)[0, 0]  # [C,H,W] after squeeze

        for c in range(num_classes):                                           # Loop through classes
            ax = plt.subplot(B, n_cols, t * n_cols + 3 + c)                    # Place subplot for class c
            ax.imshow(one_hot[c].cpu().numpy(), cmap='gray')                   # Show class channel
            ax.set_title(f"Class {c}")                                         # Title
            ax.axis('off')                                                     # Hide axes

    plt.tight_layout()                                                         # Adjust spacing
    plt.show()                                                                 # Render



def show_slices_scribble(batch, volume_idx: int,
                         slice_indices,
                         num_classes: int, device):
    """
    Visualise multiple spatial slices for ONE time-frame (volume_idx).

    Parameters:
        batch         : dict with keys 'image', 'mask', 'scribble', 'name'
        volume_idx    : int, which time index to show (0 to B-1)
        slice_indices : list of slice indices (subset of [0,…,S-1])
        num_classes   : int, number of classes in segmentation
        device        : torch.device, used for one-hot encoding
    """
    imgs  = batch['image']     # [B, S, 1, H, W] images
    masks = batch['mask']      # [B, S, 1, H, W] labels
    scrs  = batch['scribble']  # [B, S, 1, H, W] scribbles

    n_rows = len(slice_indices)                 # Number of slices to visualize
    n_cols = 2 + num_classes   # Image | Image+Scribble | One-hot channels

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))  # Configure figure

    for r, s_idx in enumerate(slice_indices):
        # --- Panel 1: Echo image ---
        img = imgs[volume_idx, s_idx, 0].cpu().numpy()                         # Extract image slice
        img_vis = np.clip(255 * (img * 0.5 + 0.5), 0, 255).astype(np.uint8)    # De-normalize for display

        ax = plt.subplot(n_rows, n_cols, r * n_cols + 1)                       # Place subplot (row r, col 1)
        ax.imshow(img_vis, cmap='gray')                                        # Show image
        ax.set_title(f"Slice {s_idx:02d} | Image")                              # Title with slice index
        ax.axis('off')                                                         # Hide axes

        # --- Panel 2: Image + Scribble ---
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)                    # Convert to color

        s = scrs[volume_idx, s_idx, 0].cpu().numpy()                           # Scribble mask
        s_bin = (s > 0).astype(np.uint8)                                       # Binary version
        s_vis = cv2.dilate(s_bin, np.ones((3, 3), np.uint8), iterations=1)     # Dilate for visibility
        img_rgb[s_vis > 0] = (0, 0, 255)                                       # Overlay red scribble

        ax = plt.subplot(n_rows, n_cols, r * n_cols + 2)                       # Place subplot (row r, col 2)
        ax.imshow(img_rgb)                                                     # Display overlay
        ax.set_title("Image + Scribble")                                       # Title
        ax.axis('off')                                                         # Hide axes

        # --- Panels 3…: One-hot channels ---
        m_slice = masks[volume_idx, s_idx, 0][None, None, None]  # [1,1,1,H,W] # Prepare for one-hot
        one_hot = make_one_hot(m_slice.to(device), device, num_classes)[0, 0]  # [C,H,W]

        for c in range(num_classes):                                           # Iterate classes
            ax = plt.subplot(n_rows, n_cols, r * n_cols + 3 + c)               # Place subplot
            ax.imshow(one_hot[c].cpu().numpy(), cmap='gray')                   # Show class c map
            ax.set_title(f"Class {c}")                                         # Title
            ax.axis('off')                                                     # Hide axes

    plt.tight_layout()                                                         # Adjust layout
    plt.show()                                                                 # Render
