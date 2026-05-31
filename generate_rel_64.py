"""
Generate smplx_verts_id_uniform_ds_rel_64.pt

This file stores 64 indices INTO the existing 698-point downsampled set
(NOT into the full 10475 SMPL-X mesh). It's used at dataset.py:144

    bps_sbj[:, self.rel_verts_idx_list]

to subset the (T, 698, 3) BPS feature to (T, 64, 3) before feeding the model.

Strategy: Farthest Point Sampling (FPS) from 698 -> 64 in T-pose.
Why FPS:
  - Same algorithm originally used for 10475 -> 698 (preprocess_dataset.py:96).
    Keeping the same selection criterion makes the 64 points a true uniform
    subset of the 698, preserving spatial coverage.
  - Maximally spreads the 64 points across the body surface, so each major
    body region (head, torso, arms, hands, hips, legs, feet) gets at least
    a few representatives - exactly what BPS-based scene interaction needs.
  - Deterministic given the T-pose; no learned/handcrafted heuristics.
"""
import os
import torch
import numpy as np
import smplx
import open3d as o3d

BODY_MODELS_PATH = os.environ.get('BODY_MODELS_PATH', './body_models/')
UNIFORM_15_PATH  = 'smplx_verts_id_uniform_ds_15.pt'
OUT_PATH         = 'smplx_verts_id_uniform_ds_rel_64.pt'
N_OUT            = 64
SEED             = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Load SMPL-X in T-pose to get the 10475 canonical vertex positions
sbj_m = smplx.create(
    model_path=BODY_MODELS_PATH, model_type='smplx',
    gender="neutral", use_pca=False, flat_hand_mean=True, batch_size=1,
).cpu().eval()

with torch.no_grad():
    tpose = sbj_m(body_pose=torch.zeros(1, 63),
                  transl=torch.zeros(1, 3),
                  global_orient=torch.zeros(1, 3))
verts_10475 = tpose.vertices.reshape(-1, 3).detach().cpu().numpy()
print(f"SMPL-X T-pose vertices: {verts_10475.shape}")

# 2. Load the 698 uniformly-downsampled indices
uniform_15 = torch.load(UNIFORM_15_PATH).tolist()
verts_698 = verts_10475[uniform_15]  # (698, 3)
print(f"Loaded {len(uniform_15)} uniform indices -> verts_698: {verts_698.shape}")

# 3. FPS from 698 -> 64
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(verts_698)
sub_pcd = pcd.farthest_point_down_sample(N_OUT)
sub_points = np.asarray(sub_pcd.points)
print(f"FPS produced {sub_points.shape[0]} points")

# 4. Map each FPS point back to its index within the 698
rel_idx = []
for p in sub_points:
    i = int(np.linalg.norm(verts_698 - p, axis=1).argmin())
    rel_idx.append(i)
rel_idx = sorted(set(rel_idx))  # dedup + sorted
print(f"Unique rel indices: {len(rel_idx)}")

# 5. Save
torch.save(torch.tensor(rel_idx, dtype=torch.long), OUT_PATH)
print(f"Saved -> {OUT_PATH}")

# 6. Sanity report - which body parts are covered
selected_verts = verts_698[rel_idx]
y_min, y_max = selected_verts[:, 1].min(), selected_verts[:, 1].max()
print(f"\nBody-Y span of selected verts: [{y_min:.3f}, {y_max:.3f}]  "
      f"(SMPL-X T-pose ranges ~-1.2 (feet) .. +0.7 (head))")
print("Y-bin distribution (foot -> head):")
bins = np.linspace(y_min, y_max, 9)
hist, _ = np.histogram(selected_verts[:, 1], bins=bins)
for i, h in enumerate(hist):
    print(f"  bin {i} [{bins[i]:+.2f} .. {bins[i+1]:+.2f}] : {'#'*h} ({h})")
