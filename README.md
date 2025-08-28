# 🫀 FeEcho4D: 4D Reconstruction of Fetal Left Ventricle from Echocardiography

# ⚠ More details will be released soon.

**Official repository for:**  
**4D Reconstruction of Fetal Left Ventricle from Echocardiography via 2.5D Radial Segmentation and Graph-Fourier Reconstruction**  
Md. Kamrul Hasan†, Qifeng Wang†, Haziq Shahard, Lucas Iijima, Nida Ruseckaite, Iris Scharnreitner, Andreas Tulzer, Bin Liu, Guang Yang‡, Choon Hwai Yap‡  

---

## 📌 Overview

This repository provides a complete pipeline for **4D fetal cardiac reconstruction** from echocardiography. Our pipeline introduces:

- **FeEcho4D**: The first benchmark dataset for radial fetal echocardiography.
- **SCOPE-Net**: A novel geometry-aware segmentation network.
- **Graph-Fourier Mesh Reconstruction**: High-fidelity reconstruction from sparse radial slices.
- **Clinical Evaluation**: Ejection Fraction (EF), Global Longitudinal/Circumferential Strain (GLS/GCS).

---

## 🧭 Pipeline Overview
We propose a three-stage framework for 4D fetal LV analysis:
-	**(A)**	Radial Data Preparation: Extract 2D slices by rotating planes around the LV center.
-	**(B)**	SCOPE-Net Segmentation: Perform prompt-guided, symmetry-aware segmentation on radial slices.
-	**(C)**	3D Mesh Reconstruction: Reconstruct temporally consistent 3D LV meshes using GHD + DVS, enabling clinical metric estimation.
<p align="center">
  <img src="assets/pipeline_overview.png" alt="Pipeline Overview" width="888"/>
</p>

---
## Radial Slice Construction from 4D Echocardiography

Given a 4D sequence \( V \in \mathbb{R}^{T \times H' \times W' \times D'} \), this script:

1. **Selects** an axial reference plane index \( Z_m \) (user-chosen slice index in depth \( D' \)).
2. **Uses** anisotropic voxel spacing \([s_x, s_y, s_z]\) from `scale.txt`.
3. **Builds** a regular 3D meshgrid \( X \) over the rotated volume at each time \( t \).
4. **Recenters** coordinates to the selected LV center \([C_x, C_y]\) (in pixels) and \( Z_m \).
5. **Applies** uniform angular sampling \( \theta \in [0, \pi] \) at \(5^\circ\) increments  
   → \( S = 37 \) slices: \( \theta = \pi \cdot s / (S-1), \ s = 0, \ldots, S{-}1 \).
6. **Rotates** coordinates via \( R_y(\theta) \) and maps back to the original frame.
7. **Interpolates** cubic 3D values at transformed coordinates to obtain \( V^t_\theta \).
8. **Extracts** the axial slice at \( Z_m \) from \( V^t_\theta \) ⇒ \( I^t_\theta \in \mathbb{R}^{H \times W} \).
9. **Stacks** all angles to form \( V^t_\theta \in \mathbb{R}^{S \times H \times W} \);  
   over \( t \), \( V_\theta \in \mathbb{R}^{T \times S \times H \times W} \).

---

### Notes on Code-to-Math Mapping
- `spacing_values = [s_x, s_y, s_z]` → read from `scale.txt`
- `lv_center = [C_x, C_y]` → LV center in pixels (rotated & cropped frame)
- `lv_start_end_frame` → acts as \( Z_m \) (axial slice index)
- Angular loop `0:5:180` → produces \( S = 37 \) slices, including 0° and 180°
- Rotation axis = **y** → consistent with \( R_y(\theta) \) in the paper

---

## 🧠 SCOPE-Net: Symmetry-Aware Prompt-Guided Segmentation

**SCOPE-Net is designed specifically for radial fetal ultrasound. It integrates:**
-	Flip-Consistent Radial Attention (FCRA) for angular symmetry modeling.
-	Inter-Slice Augmentation Invariance (ISAI) for self-supervised consistency.
-	Prompt Conditioning using bounding box or scribble inputs.
-	Efficient 2.5D training with 56G FLOPs per frame (vs. 79G for 3D UNet).

**Architecture Highlights:**
-	U-Net backbone with symmetry-aware modules.
-	Optional spatial prompts injected via gating.
-	Robust to radial view variations and signal dropout.

<p align="center">
  <img src="assets/SCOPENet.png" alt="Pipeline Overview" width="888"/>
</p>

---

## 📂 FeEcho4D Dataset

**FeEcho4D is the first public dataset for 4D radial fetal echocardiography.**
-	🧪 52 subjects, 1,845 annotated 3D volumes, 3M+ annotated 2D slices
-	🌀 37 radial views per volume, full 4D coverage
-	🎯 Manual annotation across the full cardiac cycle, including both ED and ES frames
- ✅ Clinical metrics: EF, GLS, GCS, EDV, ESV, SV

**📎 Access the dataset and tools:**
👉 [**FeEcho4D**](https://feecho4d.github.io/Website/)

---

## GHD-based 3D Mesh Reconstruction

Given a sequence of 3D segmentation volumes V \in \mathbb{R}^{T \times H \times W \times D}, the pipeline reconstructs a continuous left-ventricle (LV) mesh by Graph Harmonic Deformation (GHD):
	1.	Initialize a canonical template mesh M_0 (e.g., a sphere or averaged LV shape).
	2.	Embed vertices \{v_i\}_{i=1}^N into a graph structure with Laplacian basis functions.
	3.	Load voxel-wise segmentation masks (binary myocardium/ventricle) and anisotropic voxel spacing.
	4.	Voxelize & Sample: obtain point clouds from the mask boundary at each time t.
	5.	Fit: deform the template mesh M_0 to match sampled boundary points using the GHD energy:
    E = E_{\text{data}} + \lambda_{\text{cot}} E_{\text{cotlap}} + \lambda_{\text{dis}} E_{\text{dislap}} + \lambda_{\text{std}} E_{\text{stdlap}}
	   •	E_{\text{data}}: point-to-surface alignment (data term)
	   •	E_{\text{cotlap}}, E_{\text{dislap}}, E_{\text{stdlap}}: Laplacian regularizers
	6.	Optimize coefficients in harmonic space (low-dimensional basis) for efficient deformation.
	7.	Iterate over all time frames to produce smooth temporal mesh sequence \{M_t\}_{t=1}^T.
	8.	Output reconstructed meshes in .obj format under each case directory.

To perform parametric 3D mesh fitting using GHD on fetal cardiac masks, follow the two-step process:

Step 1: Quickstart via Jupyter Notebook
```bash
# Step into the Part(C)GHD folder (if not already there)
cd /path/to/Part(C)GHD

# Launch the notebook for interactive fitting
jupyter notebook ghd_fit_quickstart.ipynb
```
Step 2: Advanced Execution via Python Script
```bash
# Explore ghd_fit.py for full parameter control and customization
python ghd_fit.py \
    --data_root data_example \
    --cases FeEcho4D_017 \
    --times time001-010 \
    --device cuda:0 \
    --mesh_out meshes_out \
    --myo_idx 2
```

---

## ⚕️ Clinical Evaluation & Results

<p align="center">
  <img src="assets/PointCloud.png" alt="Pipeline Overview" width="888"/>
</p>

**🔍 Experiment:** We compare point clouds between predicted and ground-truth meshes in both short-axis views and 3D perspectives on FeEcho4D and MITEA, using SCOPE-Net vs. UNet.
**✅ Summary:** SCOPE-Net shows superior spatial alignment, especially at the apex and lateral wall, indicating better segmentation consistency and reconstruction quality.

<p align="center">
  <img src="assets/Clincal.png" alt="Pipeline Overview" width="888"/>
</p>

**🔍 Experiment:** On the MITEA dataset, we evaluate clinical metrics (EF, GLS) predicted by SCOPE-Net+GHD, UNet+GHD, and 3D UNet, reporting Pearson r, MSE, and 95% confidence intervals.
**✅ Summary:** Our method achieves the highest accuracy and lowest variance, demonstrating strong potential for reliable clinical use in fetal cardiac analysis.

---
## 📈 Citation

If you find this work helpful, please cite:

```bibtex
@article{hasan2025feecho4d,
  title={4D Reconstruction of Fetal Left Ventricle from Echocardiography via 2.5D Radial Segmentation and Graph-Fourier Reconstruction},
  author={XXX},
  journal={XXX},
  volume={XXX},
  pages={XXX},
  year={2025},
  doi={XXX}
}
```
---
## 🙏 Acknowledgements

- 👏We thank all co-authors for their contributions to this work, particularly in model development, dataset construction, and clinical validation. 
- 👏 Special thanks to Kepler University Hospital for their support in data acquisition and expert annotations.
- 👏And to Imperial College London and Dalian University of Technology for providing research infrastructure and technical guidance.
-  [[Wecome to Qifeng's Github]](https://github.com/QifengWang0702) [[Wecome to Haziq's Github]](https://github.com/haziqshahard) [[Wecome to Yihao's Github]](https://github.com/Luo-Yihao)
