import argparse
import json
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model


def parse_args():
    p = argparse.ArgumentParser("C-shape XML -> OCC -> SceneMI demo")
    p.add_argument("--ckpt", type=str, default="save/diffusion_scenemib/model000250000.pt")
    p.add_argument("--device", type=int, default=-1, help="-1 for CPU")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="demo_outputs/c_traverse")
    p.add_argument("--skip_timesteps", type=int, default=0, help=">0 to speed up sampling")
    return p.parse_args()


def load_args_from_ckpt(ckpt_path: Path):
    with open(ckpt_path.parent / "args.json", "r") as f:
        return SimpleNamespace(**json.load(f))


def write_c_xml(xml_path: Path):
    root = ET.Element("scene", name="C_shape")
    meta = ET.SubElement(root, "meta")
    meta.set("unit", "meter")
    # World range used for voxelization: x,z in [0, 4], y in [0, 2]
    # C shape made by 3 bars (top, left, bottom), right side open.
    walls = ET.SubElement(root, "walls")
    # Each wall: axis-aligned box with min/max in x,y,z.
    specs = [
        ("top",    0.6, 3.2, 0.0, 1.8, 2.9, 3.4),
        ("left",   0.6, 1.1, 0.0, 1.8, 0.6, 3.4),
        ("bottom", 0.6, 3.2, 0.0, 1.8, 0.6, 1.1),
    ]
    for name, xmin, xmax, ymin, ymax, zmin, zmax in specs:
        e = ET.SubElement(walls, "box", name=name)
        e.set("xmin", str(xmin))
        e.set("xmax", str(xmax))
        e.set("ymin", str(ymin))
        e.set("ymax", str(ymax))
        e.set("zmin", str(zmin))
        e.set("zmax", str(zmax))
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def xml_to_occ(xml_path: Path, out_npz: Path, scene_size=48, scene_channels=24):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    occ = np.zeros((scene_size, scene_channels, scene_size), dtype=np.float32)  # [X, Y, Z]

    x_min, x_max = 0.0, 4.0
    y_min, y_max = 0.0, 2.0
    z_min, z_max = 0.0, 4.0

    def to_idx(v, lo, hi, n):
        v = max(lo, min(hi, v))
        return int(round((v - lo) / (hi - lo) * (n - 1)))

    for box in root.findall("./walls/box"):
        xmin = float(box.get("xmin"))
        xmax = float(box.get("xmax"))
        ymin = float(box.get("ymin"))
        ymax = float(box.get("ymax"))
        zmin = float(box.get("zmin"))
        zmax = float(box.get("zmax"))

        xi0, xi1 = sorted([to_idx(xmin, x_min, x_max, scene_size), to_idx(xmax, x_min, x_max, scene_size)])
        yi0, yi1 = sorted([to_idx(ymin, y_min, y_max, scene_channels), to_idx(ymax, y_min, y_max, scene_channels)])
        zi0, zi1 = sorted([to_idx(zmin, z_min, z_max, scene_size), to_idx(zmax, z_min, z_max, scene_size)])
        occ[xi0:xi1 + 1, yi0:yi1 + 1, zi0:zi1 + 1] = 1.0

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, occ_scene=occ)


def build_scene_info(scene_info_path: Path, xml_path: Path):
    scene_info_path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "name": str(xml_path),
        "noise_level": 0.0,
        "type": "synthetic_c_shape",
    }
    with open(scene_info_path, "wb") as f:
        pickle.dump(info, f)


def pick_batch(loader, idx):
    for i, (motion, cond) in enumerate(loader):
        if i == idx:
            return motion, cond
    raise IndexError(idx)


def main():
    args_cli = parse_args()
    fixseed(args_cli.seed)

    out_dir = Path(args_cli.out_dir).resolve()
    scene_dir = out_dir / "synthetic_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    xml_path = scene_dir / "c_scene.xml"
    occ_path = scene_dir / "occ_scene.npz"
    scene_info_path = scene_dir / "scene_info.pickle"
    write_c_xml(xml_path)
    xml_to_occ(xml_path, occ_path, scene_size=48, scene_channels=24)
    build_scene_info(scene_info_path, xml_path)

    ckpt = Path(args_cli.ckpt).resolve()
    args = load_args_from_ckpt(ckpt)
    args.device = args_cli.device
    dist_util.setup_dist(args.device)
    dev = dist_util.dev()

    conf = DatasetConfig(
        name=args.dataset,
        batch_size=1,
        num_frames=args.num_frames,
        split=args_cli.split,
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
    loader = get_dataset_loader(conf, shuffle=False, num_workers=0, drop_last=False)

    model, diffusion = create_model_and_diffusion(args, loader)
    load_saved_model(model, str(ckpt), use_avg=True)
    model.to(dev)
    model.eval()
    diffusion.data_inv_transform_fn = (
        loader.dataset.scene_dataset.inv_transform_cuda if dev.type == "cuda"
        else loader.dataset.scene_dataset.inv_transform
    )

    motion, cond = pick_batch(loader, args_cli.sample_index)
    motion = motion.to(dev)
    cond["y"] = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cond["y"].items()}
    cond["y"]["sampling"] = True
    cond["y"]["uncond"] = False
    cond["y"]["debug"] = False

    # Replace scene condition by synthetic C-shape OCC.
    occ_raw = np.load(occ_path)["occ_scene"].astype(np.float32)  # [48,24,48]
    occ_model = torch.from_numpy(occ_raw).to(dev)
    occ_model = occ_model * 2.0 - 1.0
    occ_model = occ_model.permute(1, 0, 2).unsqueeze(0)  # [1,24,48,48]
    cond["y"]["occ_map"] = occ_model

    # Start/target conditioning in transl(x,z): right-bottom -> right-top.
    # Note: transl is first 3 dims in 135-d representation.
    obs_x0 = motion.clone()
    T = obs_x0.shape[-1]
    start_x, start_z = 3.15, 0.85
    target_x, target_z = 3.15, 3.15
    obs_x0[:, 0, 0, 0] = start_x
    obs_x0[:, 2, 0, 0] = start_z
    obs_x0[:, 0, 0, T - 1] = target_x
    obs_x0[:, 2, 0, T - 1] = target_z

    obs_mask = torch.zeros_like(motion, dtype=torch.bool, device=dev)
    obs_mask[:, :, :, 0] = True
    obs_mask[:, :, :, T - 1] = True
    bps_mask = torch.zeros_like(cond["y"]["bps_sbj"], dtype=torch.bool, device=dev)
    bps_mask[:, :, :, 0] = True
    bps_mask[:, :, :, T - 1] = True

    cond["obs_x0"] = obs_x0
    cond["obs_mask"] = obs_mask
    cond["bps_sbj_mask"] = bps_mask

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            motion.shape,
            clip_denoised=True,
            model_kwargs=cond,
            progress=True,
            skip_timesteps=args_cli.skip_timesteps,
        )

    inv = loader.dataset.scene_dataset.inv_transform
    pred_denorm = inv(sample.detach().cpu().permute(0, 2, 3, 1)).numpy()
    obs_denorm = inv((obs_x0 * obs_mask.float()).detach().cpu().permute(0, 2, 3, 1)).numpy()
    gt_denorm = inv(motion.detach().cpu().permute(0, 2, 3, 1)).numpy()

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "task": "c_shape_traverse_right_bottom_to_right_top",
                "checkpoint": str(ckpt),
                "split": args_cli.split,
                "sample_index": args_cli.sample_index,
                "start_xz": [start_x, start_z],
                "target_xz": [target_x, target_z],
                "note": "this ckpt was trained with wo_scene_feature=true, so occ may not influence trajectory",
                "xml": str(xml_path),
                "occ_npz": str(occ_path),
            },
            f,
            indent=2,
        )

    np.savez_compressed(
        out_dir / "viz_input.npz",
        pred_motion=pred_denorm,
        gt_motion=gt_denorm,
        obs_input=obs_denorm,
        obs_mask=obs_mask.detach().cpu().numpy(),
        occ_map=cond["y"]["occ_map"].detach().cpu().numpy(),
        bps_sbj=cond["y"]["bps_sbj"].detach().cpu().numpy(),
        bps_sbj_mask=bps_mask.detach().cpu().numpy(),
    )

    print(f"Saved C-shape demo to: {out_dir}")
    print(f" - XML: {xml_path}")
    print(f" - OCC: {occ_path}")
    print(f" - Viz: {out_dir / 'viz_input.npz'}")


if __name__ == "__main__":
    main()
