# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: generate RIVs and BEVs for point clouds

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from utils.utils import *
from tqdm import trange
import yaml

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8

def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        xyz = np.matmul(xyz, R)
    return xyz, xyzil['i']

def get_velo(velofile):
    return data2xyzi(np.fromfile(velofile))

if __name__ == '__main__':

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))

    scan_folder = config["ri_bev_generation"]["source_scans_root"]
    dst_folder = config["ri_bev_generation"]["target_ri_bev_root"]
    fov_up = config["ri_bev_generation"]["fov_up"]
    fov_down = config["ri_bev_generation"]["fov_down"]
    proj_H = config["ri_bev_generation"]["proj_H"]
    proj_W = config["ri_bev_generation"]["proj_W"]
    range_thresh = config["ri_bev_generation"]["range_th"]
    height_thresh = config["ri_bev_generation"]["height_th"]
    print(config["ri_bev_generation"])
    # ============================================================================

    min_range = min(range_thresh)
    max_range = max(range_thresh)
    min_height = min(height_thresh)
    max_height = max(height_thresh)

    scan_paths = load_files(scan_folder)
    file_num = len(scan_paths)
    print("the number of laser scans: ", file_num)

    for idx in trange(file_num):
        current_vertex = get_velo(scan_paths[idx])[0]
        ri_bev = np.zeros((len(range_thresh)+len(height_thresh), proj_H, proj_W))
        file_name = dst_folder + "/" + str(idx).zfill(6) +".npy"
        for i in range(len(range_thresh)-1):
            nearer_bound = range_thresh[i]
            farer_bound = range_thresh[i+1]
            lower_bound = height_thresh[i]
            upper_bound = height_thresh[i+1]
            proj_range, proj_vertex, _ = range_projection(current_vertex,
                                                          fov_up=fov_up,
                                                          fov_down=fov_down,
                                                          proj_H=proj_H,
                                                          proj_W=proj_W,
                                                          max_range=max_range,
                                                          cut_range=True,
                                                          lower_bound=nearer_bound,
                                                          upper_bound=farer_bound)
            proj_bev = bev_projection(current_vertex,
                                      proj_H=proj_H,
                                      proj_W=proj_W,
                                      max_range=max_range,
                                      cut_height=True,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)

            ri_bev[int(i+1),:,:] = proj_range
            ri_bev[int(i+1+len(range_thresh)), :, :] = proj_bev

            ri_bev[0, :, :], _, _ = range_projection(current_vertex,
                                                          fov_up=fov_up,
                                                          fov_down=fov_down,
                                                          proj_H=proj_H,
                                                          proj_W=proj_W,
                                                          max_range=max_range,
                                                          cut_range=True,
                                                          lower_bound=min_range,
                                                          upper_bound=max_range)
            ri_bev[len(range_thresh), :, :] = bev_projection(current_vertex,
                                             proj_H=proj_H,
                                             proj_W=proj_W,
                                             max_range=max_range,
                                             cut_height=True,
                                             lower_bound=min_height,
                                             upper_bound=max_height)

        np.save(file_name, ri_bev)


