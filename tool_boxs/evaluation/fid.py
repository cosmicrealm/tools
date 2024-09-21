from pytorch_fid import fid_score
import torch
from pytorch_fid.inception import InceptionV3
import os
import numpy as np
def calculate_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = fid_score.compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = fid_score.compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calculate_fid(path1,path2,batchsize=1,save_status=True):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    dims = 2048
    if save_status:
        save_path1 = path1 + "_"
        save_path2 = path2 + "_"
        print(f"Calculating FID stats... {save_path1} {save_path2}")
        save_path1_npz = save_path1 + ".npz"
        save_path2_npz = save_path2 + ".npz"
        if not os.path.exists(save_path1_npz):
            fid_score.save_fid_stats([path1,save_path1], batchsize, device, dims)
        else:
            with np.load(save_path1_npz) as f:
                m1, s1 = f['mu'][:], f['sigma'][:]
        if not os.path.exists(save_path2_npz):
            fid_score.save_fid_stats([path2,save_path2], batchsize, device, dims)
        else:
            with np.load(save_path2_npz) as f:
                m2, s2 = f['mu'][:], f['sigma'][:]
        fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    else:
        fid_value = fid_score.calculate_fid_given_paths([path1, path2], batchsize, device, dims)
    return fid_value