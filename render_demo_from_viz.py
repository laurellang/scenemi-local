import argparse
from pathlib import Path

import numpy as np
import torch
import smplx

from utils.utils_transform import sixd2aa
from data_loaders.trumans.utils.paramUtil import t2m_kinematic_chain
from data_loaders.trumans.utils.plot_script import plot_3d_motion


def decode_smpl135_to_joints(motion_135: np.ndarray, device: torch.device) -> np.ndarray:
    # motion_135: [T, 135] = transl(3) + global_orient_6d(6) + body_pose_6d(21*6)
    T = motion_135.shape[0]
    x = torch.from_numpy(motion_135).float().to(device)
    transl = x[:, 0:3]
    global_orient_6d = x[:, 3:9]
    body_pose_6d = x[:, 9:].reshape(T, 21, 6)

    global_orient_aa = sixd2aa(global_orient_6d)
    body_pose_aa = sixd2aa(body_pose_6d, batch=True).reshape(T, 63)

    model = smplx.create(
        model_path="./body_models/",
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        flat_hand_mean=True,
        batch_size=T,
    ).to(device).eval()

    with torch.no_grad():
        out = model(
            global_orient=global_orient_aa,
            body_pose=body_pose_aa,
            transl=transl,
            return_verts=False,
        )
    joints = out.joints[:, :22, :].detach().cpu().numpy()  # [T, 22, 3]
    return joints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viz_npz",
        type=str,
        default="demo_outputs/scenemi_inbetween_250k/viz_input.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="demo_outputs/scenemi_inbetween_250k",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.viz_npz)
    pred = data["pred_motion"][0, 0]  # [T, 135]
    gt = data["gt_motion"][0, 0]
    obs = data["obs_input"][0, 0]
    obs_mask = data["obs_mask"][0, :, 0, :]  # [135, T]
    gt_frames = np.where(obs_mask.any(axis=0))[0].tolist()

    pred_joints = decode_smpl135_to_joints(pred, device)
    gt_joints = decode_smpl135_to_joints(gt, device)
    obs_joints = decode_smpl135_to_joints(obs, device)

    plot_3d_motion(
        str(out_dir / "pred_motion.mp4"),
        t2m_kinematic_chain,
        pred_joints,
        title="SceneMI Pred Motion",
        dataset="humanml",
        fps=args.fps,
        gt_frames=gt_frames,
    )
    plot_3d_motion(
        str(out_dir / "gt_motion.mp4"),
        t2m_kinematic_chain,
        gt_joints,
        title="SceneMI GT Motion",
        dataset="humanml",
        fps=args.fps,
        gt_frames=gt_frames,
    )
    plot_3d_motion(
        str(out_dir / "obs_input_motion.mp4"),
        t2m_kinematic_chain,
        obs_joints,
        title="SceneMI Observed Keyframes Motion",
        dataset="humanml",
        fps=args.fps,
        gt_frames=gt_frames,
    )

    np.savez_compressed(
        out_dir / "joints_for_viz.npz",
        pred_joints=pred_joints,
        gt_joints=gt_joints,
        obs_joints=obs_joints,
        gt_frames=np.array(gt_frames, dtype=np.int32),
    )
    print(f"Saved videos and joints to {out_dir}")


if __name__ == "__main__":
    main()
