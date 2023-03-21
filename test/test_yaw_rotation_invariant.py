# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: test yaw-angle invariance

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from modules.cvtnet import CVTNet
from tools.read_samples import read_one_ri_bev_from_seq, read_rotated_one_ri_bev_from_seq
from tools.utils.utils import *
import torch
import yaml
import matplotlib.pyplot as plt
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
        f1_index = img_paths_database[0][-10:-4]

        plt.figure(figsize=(8, 2))
        plt.ion()
        for deg in range(10, 360, 10):

            raw_img = read_one_ri_bev_from_seq(f1_index, self.ri_bev_database_root)
            rot_img = read_rotated_one_ri_bev_from_seq(f1_index, self.ri_bev_database_root, deg)

            self.amodel.eval()
            raw_img_des = self.amodel(raw_img)
            rot_img_des = self.amodel(rot_img)

            plt.subplot(421)
            plt.title("raw RIV")
            raw_riv_img = raw_img.cpu().detach().numpy()[0, 0, :, :]
            plt.imshow(raw_riv_img)
            plt.axis("off")
            plt.axis("off")
            plt.subplot(423)
            plt.title("raw BEV")
            raw_bev_img = raw_img.cpu().detach().numpy()[0, 5, :, :]
            plt.imshow(raw_bev_img)
            plt.axis("off")

            plt.subplot(425)
            plt.title("rotated BEV")
            rot_bev_img = rot_img.cpu().detach().numpy()[0, 5, :, :]
            plt.imshow(rot_bev_img)
            plt.axis("off")
            plt.subplot(427)
            plt.title("rotated BEV")
            rot_bev_img = rot_img.cpu().detach().numpy()[0, 5, :, :]
            plt.imshow(rot_bev_img)
            plt.axis("off")

            plt.subplot(422)
            plt.title("global descriptor from raw input")
            raw_img_des = raw_img_des.cpu().detach().numpy()
            raw_img_des = np.repeat(raw_img_des, 32, 0)
            plt.imshow(raw_img_des)
            plt.axis("off")
            plt.subplot(426)
            plt.title("global descriptor from rotated input")
            rot_img_des = rot_img_des.cpu().detach().numpy()
            rot_img_des = np.repeat(rot_img_des, 32, 0)
            plt.imshow(rot_img_des)
            plt.axis("off")

            plt.pause(0.01)
            plt.clf()




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