#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project SeqOT: https://github.com/BIT-MJY/SeqOT
# SeqOT is the sequence enhanced version of our previous work OverlapTransformer: https://github.com/haomo-ai/OverlapTransformer
# Brief: Generate ground truth file by distance


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('./utils/')
import yaml
import numpy as np
from tools.utils.utils import *

# load config ================================================================
config_filename = '../config/config_kitti.yml'
config = yaml.safe_load(open(config_filename))
calib_file = "/home/hit/sda/KITTI/data_odometry_calib/dataset/sequences/00/calib.txt"
poses_file = "/home/hit/sda/KITTI/data_odometry_poses/dataset/poses/00.txt"

T_cam_velo = load_calib(calib_file)
T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
T_velo_cam = np.linalg.inv(T_cam_velo)

# load poses
poses = load_poses(poses_file)
pose0_inv = np.linalg.inv(poses[0])

poses_new = []
for pose in poses:
    poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
poses = np.array(poses_new)
print("poses shape is: ", poses.shape[0])

print("poses 0 is: ", poses[0])

poses_database = poses
poses_query = poses


# ============================================================================

print("How many pose nodes in database: ", poses_database.shape[0])
print("How many pose nodes in query: ", poses_query.shape[0])

all_rows = []
for i in np.arange(0, poses_query.shape[0], 5):
    print(i)
    one_row = []
    currunt_pose = poses_query[i]
    for idx in range(0,poses_database.shape[0]):
        if abs(idx - i) > 100:
            reffrent_pose = poses_database[idx]
            if np.linalg.norm(reffrent_pose[:3, -1].reshape(3,) - currunt_pose[:3, -1].reshape(3,)) < 15:
                one_row.append(idx)
    all_rows.append(one_row)
    print(str(i) + " ---> ", one_row)
    print("-----------------------------")

print(len(all_rows))
all_rows_array = np.array(all_rows)
np.save("./gt_15dis.npy", all_rows_array)

