# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: read sampled range images as single input or batch input

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
import torch
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def read_one_ri_bev_from_seq(file_num, ri_bev_root):
    depth_bev_data = np.load(ri_bev_root+file_num+".npy")
    depth_bev_data_tensor = torch.from_numpy(depth_bev_data).type(torch.FloatTensor).cuda()
    depth_bev_data_tensor = torch.unsqueeze(depth_bev_data_tensor, dim=0)
    return depth_bev_data_tensor



def read_one_batch_ri_bev_from_seq(f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2,
                                  train_overlap, overlap_thresh, ri_bev_root):  # without end
    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1
    sample_batch = torch.from_numpy(np.zeros((batch_size, 10, 32, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()
    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0
    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1
            depth_bev_data_r = np.load(ri_bev_root+train_imgf2[j]+".npy")
            depth_bev_data_tensor_r = torch.from_numpy(depth_bev_data_r).type(torch.FloatTensor).cuda()
            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_bev_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_bev_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1
    return sample_batch, sample_truth, pos_num, neg_num
