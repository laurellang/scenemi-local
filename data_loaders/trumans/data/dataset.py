import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import pickle
import glob
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.paramUtil import *

from utils.utils_transform import *

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class SceneMotionDataset(data.Dataset):
    def __init__(self,
                 opt,
                 scene_enc,
                 split_file,
                 statistics,
                 mode='train',
                 id_list=None,
                 scene_list=None,
                 keyframe_mode='uniform',
                 noise=(1.0),
                 test_noise_level=None,
                 num_frames=120,
                 trunc_bps=0.0,
                 light_bps=True,
                 beta=False,
                 body_abstract="part_all",
                 scene_size=48,
                 not_scene_scale=False,
                 sub_6d_idx=None,
                 rel_verts_idx_list=None,
                 data_rep='smpl_glo',
                 ):
        self.opt = opt
        self.mode = mode
        self.scene_enc = scene_enc
        self.noise = noise
        self.num_noise_level = len(self.noise)
        self.num_frames = num_frames
        self.max_motion_length = self.num_frames + 1

        self.data_rep = data_rep
        self.trunc_bps = trunc_bps
        self.light_bps = light_bps
        self.beta = beta

        self.body_abstract = body_abstract

        self.scene_size = scene_size
        self.not_scene_scale = not_scene_scale
        self.sub_6d_idx = sub_6d_idx
        self.rel_verts_idx_list = rel_verts_idx_list

        self.mean, self.std, self.bps_sbj_mean, self.bps_sbj_std = statistics
        self.mean = self.mean.unsqueeze(0)
        self.std = self.std.unsqueeze(0)
        self.bps_sbj_mean = self.bps_sbj_mean.unsqueeze(0).unsqueeze(0)
        self.bps_sbj_std = self.bps_sbj_std.unsqueeze(0).unsqueeze(0)

        self.id_list = id_list 
        self.scene_list = scene_list

        self.keyframe_mode = keyframe_mode
        #self.get_keyframes_mask(self.keyframe_mode)

        self.test_mode = (self.mode == "test")
        self.test_noise_level = test_noise_level

    def inv_transform(self, data):
        return data * self.std + self.mean

    def inv_transform_cuda(self, data):
        return data * self.std.to("cuda") + self.mean.to("cuda")

    def transform(self, motion, bps_sbj=None,):
        motion = (motion - self.mean) / self.std

        if self.trunc_bps > 0.0:
            if bps_sbj is not None:
                bps_sbj_norm = torch.norm(bps_sbj, dim=2).unsqueeze(-1)
                bps_sbj = bps_sbj / torch.where(bps_sbj_norm > self.trunc_bps, bps_sbj_norm, self.trunc_bps)

        if bps_sbj is not None:
            bps_sbj = (bps_sbj - self.bps_sbj_mean) / self.bps_sbj_std

        return motion, bps_sbj
    
    def get_keyframes_mask(self, keyframe_mode='uniform', trans_length=3):
        if keyframe_mode == 'uniform':
            self.input_frames = [*range(0, self.max_motion_length, trans_length)]
            self.last_frame = self.input_frames[-1] + 1
            self.gt_frames = [*range(0, self.last_frame, 1)]

        n_joints = 135

        self.input_mask = torch.zeros((self.last_frame, n_joints), dtype=bool) #, device=self.device)
        self.gt_mask = torch.zeros((self.last_frame, n_joints), dtype=bool) #, device=self.device)

        self.input_mask[self.input_frames, :] = True  # set keyframes
        self.gt_mask[self.gt_frames, :] = True

    def _get_motion_tensor(self, motion_dict):
        transl = motion_dict['transl']
        orient_6d = motion_dict['global_orient_6d']
        body_pose_6d = motion_dict['body_pose_6d'][:, self.sub_6d_idx]
        # Keep motion representation consistent with n_joints=135:
        # transl(3) + global_orient_6d(6) + body_pose_6d(21*6=126).
        return torch.cat([transl, orient_6d, body_pose_6d], dim=-1)
    
    def __len__(self):
        return len(self.id_list)
    

    def __getitem__(self, idx):
        # --- 1. Data Paths and Configuration ---
        name = self.id_list[idx]
        scene_name = self.scene_list[idx]

        noise_level = self.test_noise_level if self.test_mode else random.choice(self.noise)

        # --- 2. Load and Process Motion Data ---
        gt_motion = np.load(os.path.join(name, 'body_parms_cano_gt.pickle'), allow_pickle=True)
        input_motion = np.load(os.path.join(name, 'body_parms_cano_noisy_{}.pickle'.format(noise_level)), allow_pickle=True)

        # Betas and body abstract
        betas = gt_motion.get('betas', [[[0.]]])[0]
        body_abstract = np.array([gt_motion.get('part_height', [[0.]])])
        betas_tensor = torch.from_numpy(np.array(betas)).float()
        body_abstract_tensor = torch.from_numpy(np.array(body_abstract)).float()

        # BPS data
        bps_file = f'bps_marker_sbj_{noise_level}.npy' if self.light_bps else f'bps_sbj_{noise_level}.npy'
        bps_sbj = torch.tensor(np.load(os.path.join(name, bps_file)))
        if not self.light_bps and self.rel_verts_idx_list is not None:
            bps_sbj = bps_sbj[:, self.rel_verts_idx_list]

        # Ground truth joints
        gt_joints = gt_motion.get('global_joints', [[[0.]]])
        gt_joints_tensor = torch.from_numpy(np.array(gt_joints[:self.max_motion_length])).float()
        
        gt_motion_tensor = self._get_motion_tensor(gt_motion)[:self.max_motion_length]
        input_motion_tensor = self._get_motion_tensor(input_motion)[:self.max_motion_length]

        # Normalize motion data
        input_motion_norm = (input_motion_tensor - self.mean) / self.std
        gt_motion_norm = (gt_motion_tensor - self.mean) / self.std

        # Normalize BPS data with truncation
        if self.trunc_bps > 0.0:
            bps_sbj_norm = torch.norm(bps_sbj, dim=2, keepdim=True)
            bps_sbj = bps_sbj / torch.clamp(bps_sbj_norm, min=self.trunc_bps)
        bps_sbj_norm = (bps_sbj - self.bps_sbj_mean) / self.bps_sbj_std

        # --- 3. Load and Process Scene Data ---
        occ_scene_path = os.path.join(scene_name, 'occ_scene.npz')
        if os.path.exists(occ_scene_path):
            occ_map_data = np.load(occ_scene_path)
            occ_map = torch.tensor(occ_map_data['occ_scene'])
        else:
            occ_map = torch.zeros((self.scene_size, self.scene_size, self.scene_size), dtype=torch.float32)

        if not self.not_scene_scale:
            occ_map = occ_map * 2.0 - 1.0
        occ_map = occ_map.permute(1, 0, 2)

        # --- 4. Load Scene Info and Return ---
        scene_info = np.load(os.path.join(scene_name, 'scene_info.pickle'), allow_pickle=True)
        scene_info['name'] = name
        scene_info['noise_level'] = noise_level

        return (
            input_motion_norm,
            gt_motion_norm,
            gt_joints_tensor,
            bps_sbj_norm,
            occ_map,
            scene_info,
            betas_tensor,
            body_abstract_tensor,
        )

class TRUMANS(data.Dataset):
    def __init__(self,
                 split="train",
                 datapath='./dataset/trumans_opt.txt',
                 num_frames=120,
                 scene_enc='occ',
                 not_scene_scale=False,
                 noise=(1.0),
                 test_noise_level=None,
                 data_rep='smpl_glo',
                 trunc_bps=False,
                 light_bps=True,
                 sub_bps=0,
                 beta=False,
                 body_abstract="part_height",
                 scene_size=False,
                 remove_endeffector_pose=False,
                 **kwargs):
        
        #scene_preprocess_folder = 'preprocess_trumans'   
        #data_preprocess_folder = 'preprocess_trumans'  # SET YOUR PREPROCESS FOLDER HERE

        scene_preprocess_folder = 'preprocess_120_betaFalse'
        data_preprocess_folder = 'preprocess_120_betaFalse'


        self.mode = split
        self.split = split
        self.data_rep = data_rep

        self.dataset_name = 'trumans'
        self.dataname = 'trumans'
        self.scene_enc = scene_enc
        self.noise = noise

        self.body_abstract = body_abstract

        self.trunc_bps = trunc_bps
        self.light_bps = light_bps
        self.sub_bps = sub_bps
        self.beta = beta
        self.scene_size = scene_size
        self.not_scene_scale = not_scene_scale

        self.num_frames = num_frames

        project_base_path = '.'
        dataset_opt_path = pjoin(project_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        # TODO: modernize get_opt
        opt = get_opt(dataset_opt_path, device, split, max_motion_length=num_frames)
        # get_opt already handles TRUMANS_DATA_ROOT for opt.data_root and opt.motion_dir
        opt.meta_dir = pjoin(project_base_path, opt.meta_dir)
        opt.text_dir = pjoin(project_base_path, opt.text_dir)
        opt.model_dir = pjoin(project_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(project_base_path, opt.checkpoints_dir)
        opt.save_root = pjoin(project_base_path, opt.save_root)
        opt.meta_dir = os.environ.get('TRUMANS_META_DIR', './dataset')
        self.opt = opt

        print('Loading dataset %s ...' % opt.dataset_name)
        print("mode = ", split)

        print(f"DEBUG: num_frames = {self.num_frames}") 
        #assert self.num_frames == 120

        self.split_file = pjoin(opt.data_root, f"{split}_{self.num_frames}.txt")
        
        #for real world data testing
        self.remove_endeffector_pose = remove_endeffector_pose
        if self.remove_endeffector_pose:
            self.sub_6d_idx = torch.range(6*0,6*9-1).int().tolist() + torch.range(6*11,6*21-1).int().tolist()
        else:
            self.sub_6d_idx = torch.range(6*0,6*21-1).int().tolist()
        self.rel_verts_idx_list = None
        if self.sub_bps > 0:
            rel_verts_idx_path = pjoin(opt.data_root, 'smplx_verts_id_uniform_ds_rel_{}.pt'.format(self.sub_bps))
            self.rel_verts_idx_list = torch.load(rel_verts_idx_path).int().tolist()

        
        additional_info_folder = 'preprocess_120_betaFalse'

        print("load data from here: ", self.split_file)
        print('Loading data preprocess folder %s ...' % data_preprocess_folder)

        id_list = []
        scene_list = []
        add_info_list = []
        fallback_records = []

        print(f'Loading dataset {split} from {self.split_file}...')
        with open(self.split_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("/", 2)
                if len(parts) < 3:
                    # Skip malformed lines in split files instead of crashing.
                    continue
                rel_id = parts[2]

                scene_path = pjoin(opt.data_root, scene_preprocess_folder, rel_id)
                add_info_path = pjoin(opt.data_root, additional_info_folder, rel_id, 'scene_info.pickle')
                id_path = pjoin(opt.data_root, data_preprocess_folder, rel_id)
                fallback_records.append((scene_path, add_info_path, id_path))

                if self.scene_enc == 'occ_map24':
                    scene_info_path = pjoin(opt.data_root, additional_info_folder, rel_id, 'scene_info.pickle')
                    with open(scene_info_path, 'rb') as fr:
                        verts_min_max_xz = pickle.load(fr)['verts_min_max_xz']
                    if not np.all((verts_min_max_xz >= -2.4) & (verts_min_max_xz <= 2.4)):
                        continue
                
                scene_list.append(scene_path)
                add_info_list.append(add_info_path)
                id_list.append(id_path)

        if len(id_list) == 0 and len(fallback_records) > 0:
            print("WARNING: scene filter removed all samples, fallback to unfiltered split list.")
            for scene_path, add_info_path, id_path in fallback_records:
                scene_list.append(scene_path)
                add_info_list.append(add_info_path)
                id_list.append(id_path)

        if len(id_list) == 0:
            print("WARNING: no samples parsed from split file, fallback to scanning preprocess directories.")
            pattern = pjoin(opt.data_root, data_preprocess_folder, "*", "*")
            discovered = sorted(glob.glob(pattern))
            print(f"DEBUG fallback pattern: {pattern}")
            print(f"DEBUG discovered candidate dirs: {len(discovered)}")
            for id_path in discovered:
                if not os.path.isfile(pjoin(id_path, "body_parms_cano_gt.pickle")):
                    continue
                rel_id = os.path.relpath(id_path, pjoin(opt.data_root, data_preprocess_folder))
                scene_path = pjoin(opt.data_root, scene_preprocess_folder, rel_id)
                add_info_path = pjoin(opt.data_root, additional_info_folder, rel_id, 'scene_info.pickle')
                scene_list.append(scene_path)
                add_info_list.append(add_info_path)
                id_list.append(id_path)

        print(f"Loaded {len(id_list)} valid samples from split {self.split_file}")

        mean_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_{self.data_rep}_mean.pt')
        std_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_{self.data_rep}_std.pt')
        bps_sbj_mean_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_bps_sbj_mean.pt')
        bps_sbj_std_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_bps_sbj_std.pt')

        # Load or calculate normalization statistics
        mean_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_{self.data_rep}_mean.pt')
        std_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_{self.data_rep}_std.pt')
        bps_sbj_mean_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_bps_sbj_mean.pt')
        bps_sbj_std_path = pjoin(opt.meta_dir, 'z_norm', f'{opt.dataset_name}_bps_sbj_std.pt')

        if self.mode == 'train' and not (os.path.exists(mean_path) and os.path.exists(bps_sbj_mean_path)):
            print("Calculating statistics...")
            mean, std = self._get_motion_stats(id_list)
            bps_sbj_mean, bps_sbj_std = self._get_bps_stats(id_list)
            torch.save(mean, mean_path)
            torch.save(std, std_path)
            torch.save(bps_sbj_mean, bps_sbj_mean_path)
            torch.save(bps_sbj_std, bps_sbj_std_path)
        else:
            mean = torch.load(mean_path)
            std = torch.load(std_path)
            bps_sbj_mean = torch.load(bps_sbj_mean_path)
            bps_sbj_std = torch.load(bps_sbj_std_path)

        
        self.scene_dataset = SceneMotionDataset(
            self.opt,
            self.scene_enc,
            self.split_file,
            [mean, std, bps_sbj_mean, bps_sbj_std],
            mode=split,
            id_list=id_list,
            scene_list=scene_list,
            keyframe_mode='uniform',
            noise=noise,
            test_noise_level=test_noise_level,
            num_frames=self.num_frames,
            data_rep=self.data_rep,
            trunc_bps=self.trunc_bps,
            light_bps=self.light_bps,
            beta=self.beta,
            body_abstract=self.body_abstract,
            scene_size=self.scene_size,
            not_scene_scale=self.not_scene_scale,
            sub_6d_idx=self.sub_6d_idx,
            rel_verts_idx_list=self.rel_verts_idx_list,
            )

        assert len(self.scene_dataset) > 1


    def __getitem__(self, item):
        return self.scene_dataset.__getitem__(item)

    def __len__(self):
        return self.scene_dataset.__len__()
    
    def _get_motion_stats(self, id_list):
        if len(id_list) == 0:
            raise RuntimeError(f"No valid samples found for motion stats. Check split/data path: {self.split_file}")
        motion_list = []
        for name in tqdm(id_list, desc="Calculating Motion Stats"):
            motion_dict = np.load(pjoin(name, 'body_parms_cano_gt.pickle'), allow_pickle=True)
            transl = motion_dict['transl']
            orient_6d = motion_dict.get('global_orient_6d', aa2sixd(motion_dict['global_orient']))
            body_pose_6d = motion_dict['body_pose_6d'][:, self.sub_6d_idx]

            combined_data = np.concatenate([transl, orient_6d, body_pose_6d], axis=-1)
            motion_list.append(torch.from_numpy(combined_data).float())

        if len(motion_list) == 0:
            raise RuntimeError(f"No motion tensors collected from {len(id_list)} ids. Please verify preprocess files.")
        all_motions = torch.cat(motion_list, dim=0)
        mean, std = torch.std_mean(all_motions, dim=0)
        return mean, std

    def _get_bps_stats(self, id_list):
        if len(id_list) == 0:
            raise RuntimeError(f"No valid samples found for BPS stats. Check split/data path: {self.split_file}")
        bps_list = []
        for name in tqdm(id_list, desc="Calculating BPS Stats"):
            bps_file = 'bps_marker_sbj_0.0.npy' if self.light_bps else 'bps_sbj_0.0.npy'
            bps_data = np.load(pjoin(name, bps_file))
            bps_list.append(torch.from_numpy(bps_data).float())
        
        if self.trunc_bps > 0.0:
            bps_list = [bps / bps.norm(dim=2, keepdim=True).clamp(min=self.trunc_bps) for bps in bps_list]

        if len(bps_list) == 0:
            raise RuntimeError(f"No BPS tensors collected from {len(id_list)} ids. Please verify preprocess files.")
        all_bps = torch.cat(bps_list, dim=0)
        bps_sbj_std, bps_sbj_mean = torch.std_mean(all_bps.reshape(-1, 3), dim=0)
        return bps_sbj_mean, bps_sbj_std
    
