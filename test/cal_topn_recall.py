# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: calculate top@N recall to evaluate recognition performance

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import yaml
import numpy as np
from tools.utils.utils import *


def cal_topn_recall(recog_list_path, ground_truth_file_name, ri_bev_query_root, topn):

    recog_list = np.load(recog_list_path)['arr_0']
    recog_list = np.asarray(recog_list, dtype='float32')
    recog_list = recog_list.reshape((len(recog_list), 3))

    ground_truth = np.load(ground_truth_file_name, allow_pickle='True')

    gt_pt = 0
    cnt = 0
    check_out = 0

    img_paths_query = load_files(ri_bev_query_root)
    for idx in range(0, len(img_paths_query), 5):
        gt_idxes = np.array(ground_truth[int(gt_pt)])
        if gt_idxes.any():
            cnt += 1
        else:
            gt_pt += 1
            continue
        gt_pt += 1
        recog_list_for_this_scan = recog_list[recog_list[:,0]==idx,:]
        idx_sorted = np.argsort(recog_list_for_this_scan[:,-1], axis=-1)
        for i in range(topn):
            if int(recog_list_for_this_scan[idx_sorted[i], 1]) in gt_idxes:
                check_out += 1
                break

    print("top"+str(topn)+" recall ", round(check_out/cnt, 4))
    return round(check_out/cnt, 4)


if __name__ == "__main__":

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    ri_bev_database_root = config["data_root"]["ri_bev_database_root"]
    ri_bev_query_root = config["data_root"]["ri_bev_query_root"]
    recog_list_path = config["cal_topn_recall"]["recog_list"]
    gt_path = config["cal_topn_recall"]["ground_truth"]
    topn = config["cal_topn_recall"]["topn"]
    print(config["cal_topn_recall"])
    # ============================================================================
    recall_list = []
    for i in range(1,topn+1):
        rec = cal_topn_recall(recog_list_path, gt_path, ri_bev_query_root, i)
        recall_list.append(rec)
    print("================================================")
    print("================================================")
    print("================================================")
    print(recall_list)
