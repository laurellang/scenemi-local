import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.editing_util import get_keyframes_mask
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util


def parse_args():
    parser = argparse.ArgumentParser(
        description="SceneMI demo: scene-aware motion in-betweening from sparse keyframes"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="save/diffusion_scenemib/model000250000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--device", type=int, default=-1, help="-1 means CPU")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=30,
        help="Uniform keyframe interval for in-betweening condition",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="demo_outputs/scenemi_inbetween_250k",
        help="Output directory",
    )
    return parser.parse_args()


def load_args_from_ckpt(ckpt_path: Path):
    args_json = ckpt_path.parent / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found next to ckpt: {args_json}")
    with open(args_json, "r") as f:
        d = json.load(f)
    return SimpleNamespace(**d)


def pick_batch(loader, sample_index: int):
    for i, (motion, cond) in enumerate(loader):
        if i == sample_index:
            return motion, cond
    raise IndexError(f"sample_index={sample_index} out of range for split loader")


def main():
    cli = parse_args()
    fixseed(cli.seed)

    ckpt_path = Path(cli.ckpt).resolve()
    out_dir = Path(cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args = load_args_from_ckpt(ckpt_path)
    args.device = cli.device

    dist_util.setup_dist(args.device)
    dev = dist_util.dev()

    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=1,
        num_frames=args.num_frames,
        split=cli.split,
        scene_enc=args.scene_type,
        noise=tuple(args.noise),
        data_rep=args.data_rep,
        trunc_bps=args.trunc_bps,
        light_bps=args.light_bps,
        sub_bps=args.sub_bps,
        beta=args.beta,
        body_abstract=args.body_abstract,
        scene_size=args.scene_size,
    )
    loader = get_dataset_loader(data_conf, shuffle=False, num_workers=0, drop_last=False)

    model, diffusion = create_model_and_diffusion(args, loader)
    load_saved_model(model, str(ckpt_path), use_avg=True)
    model.to(dev)
    model.eval()
    if dev.type == "cuda":
        diffusion.data_inv_transform_fn = loader.dataset.scene_dataset.inv_transform_cuda
    else:
        diffusion.data_inv_transform_fn = loader.dataset.scene_dataset.inv_transform

    motion, cond = pick_batch(loader, cli.sample_index)
    motion = motion.to(dev)
    cond["y"] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cond["y"].items()}
    cond["y"]["sampling"] = True
    cond["y"]["uncond"] = False
    cond["y"]["debug"] = False

    # Paper task: sparse keyframe constrained in-betweening in scene.
    cond["obs_x0"] = motion
    obs_mask, bps_sbj_mask = get_keyframes_mask(
        motion,
        cond,
        edit_mode="uniform",
        interval=cli.keyframe_interval,
        p1=0.0,
        p2=1.0,
    )
    cond["obs_mask"] = obs_mask
    cond["bps_sbj_mask"] = bps_sbj_mask

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            motion.shape,
            clip_denoised=True,
            model_kwargs=cond,
            progress=True,
        )

    # Denormalize motion for visualization input.
    pred_denorm = loader.dataset.scene_dataset.inv_transform(sample.detach().cpu().permute(0, 2, 3, 1))
    gt_denorm = loader.dataset.scene_dataset.inv_transform(motion.detach().cpu().permute(0, 2, 3, 1))
    obs_input_denorm = loader.dataset.scene_dataset.inv_transform(
        (cond["obs_x0"] * cond["obs_mask"].float()).detach().cpu().permute(0, 2, 3, 1)
    )

    scene_info = cond["y"]["scene_info"][0]
    meta = {
        "task": "scene_aware_motion_inbetweening",
        "split": cli.split,
        "sample_index": cli.sample_index,
        "keyframe_interval": cli.keyframe_interval,
        "checkpoint": str(ckpt_path),
        "scene_name": scene_info.get("name", "unknown"),
        "noise_level": float(scene_info.get("noise_level", -1)),
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        out_dir / "viz_input.npz",
        pred_motion=pred_denorm.numpy(),     # [1, T, D]
        gt_motion=gt_denorm.numpy(),         # [1, T, D]
        obs_input=obs_input_denorm.numpy(),  # [1, T, D]
        obs_mask=cond["obs_mask"].detach().cpu().numpy(),          # [1, D, 1, T]
        occ_map=cond["y"]["occ_map"].detach().cpu().numpy(),       # [1, 24, 48, 48]
        bps_sbj=cond["y"]["bps_sbj"].detach().cpu().numpy(),       # [1, 64, 3, T]
        bps_sbj_mask=cond["bps_sbj_mask"].detach().cpu().numpy(),  # [1, 64, 3, T]
    )

    print(f"Saved demo outputs to: {out_dir}")
    print(f"meta: {out_dir / 'meta.json'}")
    print(f"viz:  {out_dir / 'viz_input.npz'}")


if __name__ == "__main__":
    main()
