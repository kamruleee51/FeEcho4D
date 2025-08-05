


"""
GHD Fitting Script

Example
-------
python fit_ghd.py \\
       --data_root /path/to/data_example \\
       --cases FeEcho4D_017 \\
       --times time001-005 \\
       --device cuda:0 \\
       --num_iter 500 \\
       --lr_start 1e-2 \\
       --myo_idx 2
"""

# -----------------------------------------------------------------------------#
# Imports
# -----------------------------------------------------------------------------#
import argparse
import re
from pathlib import Path
from typing import Sequence, Set

import numpy as np
import torch
import trimesh

# --------------------------- project-specific --------------------------------#
from GHD.GHD_cardiac import GHD_Cardiac
from GHD import GHD_config
from data_process.dataset_real_scaling import *
# -----------------------------------------------------------------------------#


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def parse_time_tokens(tokens: Sequence[str]) -> Set[str]:
    """
    Expand user-provided time strings.

    Examples
    --------
    ['ED', 'ES', 'time001-003'] → {'ED', 'ES', 'time001', 'time002', 'time003'}
    """
    sel: Set[str] = set()
    pat_single = re.compile(r"^(ED|ES|time\d{3})$")
    pat_range  = re.compile(r"^(time)(\d{3})-(\d{3})$")

    for t in tokens:
        if pat_single.fullmatch(t):
            sel.add(t)
        elif (m := pat_range.fullmatch(t)):
            prefix, s, e = m.groups()
            for i in range(int(s), int(e) + 1):
                sel.add(f"{prefix}{i:03d}")
        else:
            raise ValueError(f"Unrecognised time token: {t}")
    return sel


def guess_time_tag(fname: str) -> str:
    """Return 'ED', 'ES', or 'timeXYZ' from a file name."""
    if (m := re.search(r"(ED|ES|time\d{3})", fname)):
        return m.group(1)
    raise RuntimeError(f"Cannot extract time tag from '{fname}'")

# ------------------------------------------------------------------------------
# Core per-case routine
# ------------------------------------------------------------------------------
def process_case(
    case_path: Path,
    args: argparse.Namespace,
    cfg: GHD_config,
    time_filters: Set[str],
) -> None:
    """Fit one case (all requested frames) and save OBJ meshes."""
    mesh_out_case = args.mesh_out / case_path.name
    mesh_out_case.mkdir(parents=True, exist_ok=True)

    for mask_type in args.mask_types:
        nii_dir = args.nifti_root / case_path.name / mask_type
        if not nii_dir.exists():
            print(f"⚠  {nii_dir} not found – skipped")
            continue

        nii_files = sorted(nii_dir.glob("*.nii")) + sorted(nii_dir.glob("*.nii.gz"))
        if not nii_files:
            print(f"⚠  No NIfTI files in {nii_dir} – skipped")
            continue

        dataset = Fetal_motion_tracked(nii_dir)

        for idx, nii_path in enumerate(nii_files):
            tag = guess_time_tag(nii_path.name)
            if time_filters and tag not in time_filters:
                continue

            print(f"🌟  {case_path.name} | {tag} | {mask_type}")

            sample     = dataset[idx]
            mask_tensor = sample[next(iter(sample))].to(args.device)
            unique_vals = torch.unique(mask_tensor)

            # ------------------------------------------------------------------
            # Point-cloud extraction (labels order: 0, ENDO, MYO)
            # ------------------------------------------------------------------
            labels = [0, unique_vals[1], unique_vals[2]]  # relies on sorted values
            pts    = point_cloud_extractor(
                mask_tensor,
                labels,
                sample["window"],
                spacing=args.pc_spacing,
                coordinate_order="zyx",
            )
            pts = [p.to(args.device) for p in pts]

            points_myo = pts[args.myo_idx]
            if points_myo.shape[0] < args.mesh_samples:
                print(f"⚠  Only {points_myo.shape[0]} MYO points "
                      f"(<{args.mesh_samples}) – skipped")
                continue

            sel = np.random.choice(points_myo.shape[0],
                                   args.mesh_samples,
                                   replace=False)
            mesh_gt_lv_sample = points_myo.detach().cpu().numpy()[sel]

            # ------------------------------------------------------------------
            # GHD fitting
            # ------------------------------------------------------------------
            heart = GHD_Cardiac(cfg)
            _ = heart.rendering()                       # initialize mesh
            heart.global_registration_lv(mesh_gt_lv_sample)

            loss_w = dict(
                Loss_occupancy=args.loss_occupancy,
                Loss_normal_consistency=args.loss_normal_consistency,
                Loss_Laplacian=args.loss_laplacian,
                Loss_thickness=args.loss_thickness,
            )

            heart.morphing2lvtarget(
                points_myo.float(),
                torch.empty(0, 3, device=args.device),  # no extra points
                target_mesh=None,
                loss_dict=loss_w,
                lr_start=args.lr_start,
                num_iter=args.num_iter,
                num_sample=args.num_sample,
                NP_ratio=args.NP_ratio,
                if_reset=True,
                if_fit_R=True,
                if_fit_s=True,
                if_fit_T=True,
                record_convergence=True,
            )

            fitted = heart.rendering()
            out_f  = mesh_out_case / f"{case_path.name}_{tag}_{mask_type}.obj"
            trimesh.Trimesh(
                vertices=fitted.verts_packed().cpu().numpy(),
                faces=fitted.faces_packed().cpu().numpy(),
            ).export(out_f)
            print(f"✅  Saved → {out_f}")

            del heart, fitted, pts, points_myo
            torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def run(raw_args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit GHD parametric meshes to fetal cardiac MRI masks"
    )

    # --- Paths ---------------------------------------------------------------
    parser.add_argument("--data_root", type=Path, required=True,
        help="Folder containing one sub-folder per case.")
    parser.add_argument("--nifti_root", type=Path,
        help="If masks are stored elsewhere, override here "
             "(default = data_root).")
    parser.add_argument("--mesh_out", type=Path, default=Path("meshes_out"),
        help="Where to save generated OBJ meshes.")

    # --- Selection -----------------------------------------------------------
    parser.add_argument("--cases", nargs="*",
        help="Limit to these case names (default: all).")
    parser.add_argument("--times", nargs="*",
        help="ED / ES / timeXYZ tokens; ranges like time001-005 are allowed.")

    # --- Dataset layout ------------------------------------------------------
    parser.add_argument("--mask_types", nargs="+", default=["mask"],
        help="Sub-folder(s) inside each case that hold the NIfTI masks.")
    parser.add_argument("--myo_idx", type=int, choices=[1, 2], default=2,
        help="Which entry in point_list is myocardium "
             "(FeEcho4D=2, MITEA=1).")

    # --- Optimiser hyper-parameters -----------------------------------------
    parser.add_argument("--num_iter", type=int, default=500,
        help="#Gradient steps for morphing.")
    parser.add_argument("--num_sample", type=int, default=30_000,
        help="#Point-cloud samples per step.")
    parser.add_argument("--NP_ratio", type=int, default=2,
        help="Negative/positive sample ratio for occupancy loss.")
    parser.add_argument("--lr_start", type=float, default=1e-2,
        help="Initial learning rate.")
    parser.add_argument("--mesh_samples", type=int, default=4_000,
        help="#Surface points to initialise global registration.")

    # --- Loss weights --------------------------------------------------------
    parser.add_argument("--loss_occupancy",          type=float, default=1.0)
    parser.add_argument("--loss_normal_consistency", type=float, default=0.01)
    parser.add_argument("--loss_laplacian",          type=float, default=0.01)
    parser.add_argument("--loss_thickness",          type=float, default=0.01)

    # --- Misc ----------------------------------------------------------------
    parser.add_argument("--pc_spacing", type=float, default=200,
        help="Voxel spacing (µm) passed to point_cloud_extractor.")
    parser.add_argument("--device", default="cuda:0",
        help="PyTorch device string, e.g. 'cuda:0' or 'cpu'.")

    args = parser.parse_args(raw_args)

    # Derived paths
    args.nifti_root = args.nifti_root or args.data_root
    args.mesh_out.mkdir(parents=True, exist_ok=True)

    # GHD configuration
    cfg = GHD_config(
        base_shape_path=Path(__file__).resolve().parent /
                        "canonical_shapes/Standard_LV_2000_fetal256.obj",
        num_basis=36,  # 6**2
        mix_laplacian_tradeoff=dict(cotlap=0.1, dislap=1.0, stdlap=0.1),
        device=args.device,
        if_nomalize=True,
        if_return_scipy=True,
        bi_ventricle_path=Path(__file__).resolve().parent /
                          "canonical_shapes/Standard_BiV.obj",
    )

    # Case list
    cases = [p for p in args.data_root.iterdir() if p.is_dir()]
    if args.cases:
        cases = [p for p in cases if p.name in args.cases]
    print(f"Found {len(cases)} case(s)")

    time_filters = parse_time_tokens(args.times) if args.times else set()

    # Main loop
    for c in cases:
        process_case(c, args, cfg, time_filters)


if __name__ == "__main__":
    run()