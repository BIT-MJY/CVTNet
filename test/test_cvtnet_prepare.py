# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: generate predictions for the final evaluation

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from modules.cvtnet import CVTNet
from tools.read_samples import read_one_ri_bev_from_seq
from tools.utils.utils import *
import torch
from tqdm import tqdm
import faiss
import yaml
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

class testHandler():
    def __init__(self, channels=1, ri_bev_database_root=None, ri_bev_query_root=None, pretrained_weights=None, topn=50, save_results_flag=True):
        super(testHandler, self).__init__()

        self.channels = channels
        self.amodel = CVTNet(channels=self.channels, use_transformer=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters  = self.amodel.parameters()

        self.ri_bev_database_root = ri_bev_database_root
        self.ri_bev_query_root = ri_bev_query_root
        self.pretrained_weights = pretrained_weights
        self.topn = topn
        self.save_results_flag = save_results_flag

    def eval(self):

        print("loading ", self.pretrained_weights)
        checkpoint = torch.load(self.pretrained_weights)
        self.amodel.load_state_dict(checkpoint['state_dict'])

        img_paths_database = load_files(self.ri_bev_database_root)
        des_list = np.zeros((len(img_paths_database), 768))

        for j in tqdm(range(0, len(img_paths_database))):
            f1_index = str(j).zfill(6)
            current_batch = read_one_ri_bev_from_seq(f1_index, self.ri_bev_database_root)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list[(j), :] = current_batch_des[0, :].cpu().detach().numpy()
        des_list = des_list.astype('float32')

        img_paths_query = load_files(self.ri_bev_query_root)

        nlist = 1
        d = 768
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained
        index.train(des_list)
        assert index.is_trained
        index.add(des_list)
        recog_list = []

        for i in range(0, len(img_paths_query), 5):

            i_index = str(i).zfill(6)
            current_batch = read_one_ri_bev_from_seq(i_index, self.ri_bev_query_root)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch)
            des_list_current = current_batch_des[0, :].cpu().detach().numpy()

            D, I = index.search(des_list_current.reshape(1, -1), self.topn)

            for j in range(D.shape[1]):
                one_recog = np.zeros((1,3))
                one_recog[:, 0] = i
                one_recog[:, 1] = I[:,j]
                one_recog[:, 2] = D[:,j]
                recog_list.append(one_recog)
                print("query:"+str(i) + "---->" + "database:" + str(I[:, j]) + "  " + str(D[:, j]))

        if self.save_results_flag:
            np.savez_compressed("./pr_results", np.array(recog_list))

if __name__ == '__main__':

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    ri_bev_database_root = config["data_root"]["ri_bev_database_root"]
    ri_bev_query_root = config["data_root"]["ri_bev_query_root"]
    pretrained_weights = config["test_cvtnet_prepare"]["weights"]
    num_layers = config["test_cvtnet_prepare"]["num_layers"]
    topn = config["test_cvtnet_prepare"]["topn"]
    save_results = config["test_cvtnet_prepare"]["save_results"]
    print(config["test_cvtnet_prepare"])
    # ============================================================================

    test_handler = testHandler(channels=num_layers, ri_bev_database_root=ri_bev_database_root, ri_bev_query_root=ri_bev_query_root,
                               pretrained_weights=pretrained_weights, topn=topn, save_results_flag=save_results)
    test_handler.eval()