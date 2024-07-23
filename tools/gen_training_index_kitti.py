import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import yaml
import numpy as np
from tools.utils.com_overlap_yaw import com_overlap_yaw
from tools.utils.utils import *
from tools.utils.normalize_data import normalize_data
from tools.utils.split_train_val import split_train_val

# load config ================================================================
config_filename = '../config/config_kitti.yml'

config = yaml.safe_load(open(config_filename))
scan_folder = config["ri_bev_generation"]["source_scans_root"]
train_seqs = config["ri_bev_generation"]["train_seqs"]
calib_file = "/home/hit/sda/KITTI/data_odometry_calib/dataset/sequences"
poses_file = "/home/hit/sda/KITTI/data_odometry_poses/dataset/poses"


total_training_tuple_list = []
total_index = 0

for seq in train_seqs:
    file_path = os.path.join(scan_folder, seq, "velodyne")
    print("file_path is: ", file_path)
    
    calib_file_seq = os.path.join(calib_file, seq)
    calib_file_seq = calib_file_seq + "/calib.txt"
    poses_file_seq = poses_file + "/" + str(seq) + ".txt"
    #poses_file_seq = "/home/hit/sda/SemanticKITTI/dataset/sequences/04/poses.txt"
    
    scan_paths = load_files(file_path)

    T_cam_velo = load_calib(calib_file_seq)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # load poses
    poses = load_poses(poses_file_seq)
    pose0_inv = np.linalg.inv(poses[0])

    poses_new = []
    for pose in poses:
        poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
    poses = np.array(poses_new)
    
    print("poses shape is: ", poses.shape[0])
    print("scan_paths shape is: ", len(scan_paths))
    
    div = 0
    if len(scan_paths) > 2000:
        div = 50
    else:
        div = 20
    
    training_tuple_list = []
    for i in range(len(scan_paths)):
        if i % div != 0:
            continue
        print("\nProcessing " + str(i) + "/" + str(len(scan_paths)) + "-------->")
        # calculate overlap
        scan_paths_this_frame = scan_paths[i:]
        poses_new_this_frame = poses[i:]
        # ground_truth_mapping = com_overlap_yaw(scan_paths, poses, frame_idx=0)
        ground_truth_mapping = com_overlap_yaw(scan_paths_this_frame, poses_new_this_frame, frame_idx=0, leg_output_width=360)
        print("ground_truth_mapping shape is: ", ground_truth_mapping.shape[0])
        for m in range(ground_truth_mapping.shape[0]):
            one_row = []
            idx1 = str(int(i)).zfill(6)
            idx2 = str(int(i + ground_truth_mapping[m, 1])).zfill(6)
            one_row.append(idx1)
            one_row.append(idx2)
            one_row.append(ground_truth_mapping[m, 2])
            training_tuple_list.append(one_row)
        
    
    for row in training_tuple_list:
        row[0] = str(int(row[0]) + total_index).zfill(6)
        row[1] = str(int(row[1]) + total_index).zfill(6)
    total_index += len(scan_paths)
    
    total_training_tuple_list.extend(training_tuple_list)

normalized_array = np.array(total_training_tuple_list)
output_file_path = '/home/hit/sda/place_recognition/CVTNet-main/tools/output_data.txt'
# 保存数组到文本文件 for debug
np.savetxt(output_file_path, total_training_tuple_list, fmt='%s', delimiter=',')
print("total_ground_truth_mapping shape is: ", len(total_training_tuple_list))          
        
   
no_normalized_path = os.path.join("/home/hit/sda/place_recognition/KITTI/ri_bev/overlaps")
if not os.path.exists(no_normalized_path):
    os.makedirs(no_normalized_path)
no_normalized_path_npy = no_normalized_path + "/no_normalized_data.npy"
np.save(no_normalized_path_npy, normalized_array)

################# normalize training data
no_normalize_data = np.load(no_normalized_path_npy, allow_pickle="True")
print("How many training indices: ", no_normalize_data.shape[0])

normalize_data_bin1 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<1.1) & (no_normalize_data[:,-1].astype(float)>=0.7)]
normalize_data_bin2 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.7) & (no_normalize_data[:,-1].astype(float)>=0.5)]
normalize_data_bin3 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.5) & (no_normalize_data[:,-1].astype(float)>=0.3)]
normalize_data_bin4 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.3) & (no_normalize_data[:,-1].astype(float)>=0.1)]
normalize_data_bin5 = no_normalize_data[(no_normalize_data[:,-1].astype(float)<0.1) & (no_normalize_data[:,-1].astype(float)>=0.0)]

print("1.0~0.7: ", normalize_data_bin1.shape)
print("0.7~0.5: ", normalize_data_bin2.shape)
print("0.5~0.3: ", normalize_data_bin3.shape)
print("0.3~0.1: ", normalize_data_bin4.shape)
print("0.1~0.0: ", normalize_data_bin5.shape)


min_bin_1 = 2300
min_bin_2 = 3000
min_bin_3 = 5000
min_bin_4 = 5000
min_bin_5 = 5000

chosen_idx = np.random.randint(0,normalize_data_bin1.shape[0],min_bin_1)
normalize_data_bin1_chosen = normalize_data_bin1[chosen_idx,:]

chosen_idx = np.random.randint(0,normalize_data_bin2.shape[0],min_bin_2)
normalize_data_bin2_chosen = normalize_data_bin2[chosen_idx,:]

chosen_idx = np.random.randint(0,normalize_data_bin3.shape[0],min_bin_3)
normalize_data_bin3_chosen = normalize_data_bin3[chosen_idx,:]

chosen_idx = np.random.randint(0,normalize_data_bin4.shape[0],min_bin_4)
normalize_data_bin4_chosen = normalize_data_bin4[chosen_idx,:]

chosen_idx = np.random.randint(0,normalize_data_bin5.shape[0],min_bin_5)
normalize_data_bin5_chosen = normalize_data_bin5[chosen_idx,:]


normalize_data = []
for i in range(no_normalize_data.shape[0]):
    if abs(int(no_normalize_data[i, 0]) - int(no_normalize_data[i, 1])) > 500 and no_normalize_data[i, 2].astype(float) > 0.3:
        normalize_data.append(no_normalize_data[i, :])
        continue
print("Number of long-term loop closing: ", len(normalize_data))

# min_bin = 2000
# chosen_idx = np.random.randint(0,np.array(normalize_data).shape[0],min_bin)
# normalize_data_bin6_chosen = np.array(normalize_data)[chosen_idx,:]

normalize_data_chosen = np.concatenate((normalize_data_bin1_chosen,normalize_data_bin2_chosen,normalize_data_bin3_chosen,
                                        normalize_data_bin4_chosen, normalize_data_bin5_chosen), axis=0)
print("How many training indices: ", normalize_data_chosen.shape[0])

np.save("/home/hit/sda/place_recognition/CVTNet-main/tools/train_normalized_data.npy", normalize_data_chosen)


