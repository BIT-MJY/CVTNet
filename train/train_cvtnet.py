# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: train CVTNet with database from NCLT dataset

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import torch
from tensorboardX import SummaryWriter
from modules.cvtnet import CVTNet
import modules.loss as PNV_loss
from tools.read_samples import read_one_ri_bev_from_seq
from tools.read_samples import read_one_batch_ri_bev_from_seq
import yaml
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

class trainHandler():
    def __init__(self, channels=1, lr = 0.001, step_size=5, gamma=0.9, overlap_th=0.3, use_shuffle=False,
                 num_pos=6, num_neg=6, resume=False,
                 pretrained_weights=None, save_path=None, train_set=None, ri_bev_root=None):
        super(trainHandler, self).__init__()

        self.channels = channels
        self.learning_rate = lr
        self.ri_bev_root = ri_bev_root
        self.resume = resume
        self.pretrained_weights = pretrained_weights
        self.save_path = save_path
        self.overlap_thresh = overlap_th
        self.use_shuffle = use_shuffle
        self.max_num_pos = num_pos
        self.max_num_neg = num_neg

        self.train_set = train_set
        self.train_set_imgf1_imgf2_overlap = np.load(self.train_set)

        self.amodel = CVTNet(channels=self.channels, use_transformer=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters  = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train(self):

        epochs = 50

        if self.resume:
            print("resuming from ", self.pretrained_weights)
            checkpoint = torch.load(self.pretrained_weights)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("training from scratch ..." )
            starting_epoch = 0

        writer1 = SummaryWriter(comment="LR_xxxx")

        for i in range(starting_epoch+1, epochs):

            if self.use_shuffle:
                self.train_set_imgf1_imgf2_overlap = np.random.permutation(self.train_set_imgf1_imgf2_overlap)

            self.train_imgf1 = self.train_set_imgf1_imgf2_overlap[:, 0]
            self.train_imgf2 = self.train_set_imgf1_imgf2_overlap[:, 1]
            self.train_dir1 = np.zeros((len(self.train_imgf1),))
            self.train_dir2 = np.zeros((len(self.train_imgf2),))
            self.train_overlap = self.train_set_imgf1_imgf2_overlap[:, 2].astype(float)

            print("=======================================================================\n\n\n")
            print("total pairs: ", len(self.train_imgf1))
            print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            cnt = 0
            used_list_f1 = []
            used_list_dir1 = []

            for j in range(len(self.train_imgf1)):
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                continue_flag = False
                for iddd in range(len(used_list_f1)):
                    if f1_index==used_list_f1[iddd] and dir1_index==used_list_dir1[iddd]:
                        continue_flag = True
                else:
                    used_list_f1.append(f1_index)
                    used_list_dir1.append(dir1_index)
                if continue_flag:
                    continue

                current_batch = read_one_ri_bev_from_seq(f1_index, self.ri_bev_root)
                sample_batch, sample_truth, pos_num, neg_num = read_one_batch_ri_bev_from_seq \
                    (f1_index, dir1_index, self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap,
                     self.overlap_thresh, self.ri_bev_root)

                if pos_num >= self.max_num_pos and neg_num>=self.max_num_neg:
                    sample_batch = torch.cat((sample_batch[0:self.max_num_pos, :, :, :], sample_batch[pos_num:pos_num+self.max_num_neg, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:self.max_num_pos, :], sample_truth[pos_num:pos_num+self.max_num_neg, :]), dim=0)
                    pos_num = self.max_num_pos
                    neg_num = self.max_num_neg
                elif pos_num >= self.max_num_pos:
                    sample_batch = torch.cat((sample_batch[0:self.max_num_pos, :, :, :], sample_batch[pos_num:, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:self.max_num_pos, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = self.max_num_pos
                elif neg_num >= self.max_num_neg:
                    sample_batch = sample_batch[0:pos_num+self.max_num_neg,:,:,:]
                    sample_truth = sample_truth[0:pos_num+self.max_num_neg, :]
                    neg_num = self.max_num_neg

                if neg_num == 0 or pos_num == 0:
                    continue

                input_batch = torch.cat((current_batch, sample_batch), dim=0)

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_batch)

                o1, o2, o3 = torch.split(
                    global_des, [1, pos_num, neg_num], dim=0)
                MARGIN = 0.5
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN, lazy=False)
                loss.backward()
                self.optimizer.step()
                print(str(cnt), loss)

                loss_each_epoch = loss_each_epoch + loss.item()
                cnt = cnt + 1

            print("epoch {} loss {}".format(i, loss_each_epoch/cnt))
            self.scheduler.step()
            saved_weights_name = self.save_path + "amodel_cvtnet"+str(i)+".pth.tar"
            print("saving weights to " + saved_weights_name)

            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                saved_weights_name)
            writer1.add_scalar("loss", loss_each_epoch / cnt, global_step=i)

if __name__ == '__main__':

    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    traindata_file = config["train_cvtnet"]["traindata_file"]
    num_layers = config["train_cvtnet"]["num_layers"]
    overlap_th = config["train_cvtnet"]["overlap_th"]
    learning_rate = config["train_cvtnet"]["lr"]
    step_size = config["train_cvtnet"]["step_size"]
    gamma = config["train_cvtnet"]["gamma"]
    resume = config["train_cvtnet"]["resume"]
    use_shuffle = config["train_cvtnet"]["use_shuffle"]
    num_pos = config["train_cvtnet"]["num_pos"]
    num_neg = config["train_cvtnet"]["num_neg"]
    pretrained_weights = config["train_cvtnet"]["weights"]
    save_path = config["train_cvtnet"]["save_path"]
    ri_bev_root = config["data_root"]["ri_bev_database_root"]
    print(config["train_cvtnet"])
    # ============================================================================

    train_handler = trainHandler(channels=num_layers,
                                 lr=learning_rate,
                                 step_size=step_size,
                                 gamma=gamma,
                                 overlap_th=overlap_th,
                                 use_shuffle=use_shuffle,
                                 num_pos=num_pos,
                                 num_neg=num_neg,
                                 resume=resume,
                                 pretrained_weights=pretrained_weights,
                                 save_path=save_path,
                                 train_set=traindata_file,
                                 ri_bev_root=ri_bev_root)
    train_handler.train()