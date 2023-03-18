# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: architecture of CVTNet

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import torch
import torch.nn as nn
from modules.netvlad import NetVLADLoupe
import math
import torch.nn.functional as F

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x



class featureExtracter_RI_BEV(nn.Module):
    def __init__(self, channels=5, use_transformer = True):
        super(featureExtracter_RI_BEV, self).__init__()

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(2,1), stride=(2,1), bias=False)
        self.conv1_add = nn.Conv2d(16, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(1,1), bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)

        self.relu = nn.ReLU(inplace=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)  # 3 6
        self.convLast1 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1), bias=False)
        self.convLast2 = nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, x_l):

        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv1_add(out_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))

        out_l_1 = out_l.permute(0,1,3,2)
        out_l_1 = self.relu(self.convLast1(out_l_1))

        if self.use_transformer:
            out_l = out_l_1.squeeze(3)
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)
        out_l = torch.cat((out_l_1, out_l), dim=1)
        out_l = self.relu(self.convLast2(out_l))
        return out_l




class CVTNet(nn.Module):
    def __init__(self, channels=5, use_transformer = True):
        super(CVTNet, self).__init__()

        self.use_transformer = use_transformer

        self.featureExtracter_RI = featureExtracter_RI_BEV(channels=channels, use_transformer=use_transformer)
        self.featureExtracter_BEV = featureExtracter_RI_BEV(channels=channels, use_transformer=use_transformer)

        self.relu = nn.ReLU(inplace=True)

        d_model = 256
        heads = 4
        dropout = 0.

        self.convLast2 = nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=1800, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.net_vlad_ri = NetVLADLoupe(feature_size=256, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.net_vlad_bev = NetVLADLoupe(feature_size=256, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_2_ext = Norm(d_model)
        self.norm_3_ext = Norm(d_model)

        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff1 = FeedForward(d_model, dropout=dropout)
        self.ff2 = FeedForward(d_model, dropout=dropout)

        self.attn1_ext = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2_ext = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff1_ext = FeedForward(d_model, dropout=dropout)
        self.ff2_ext = FeedForward(d_model, dropout=dropout)


    def forward(self, x_ri_bev):
        x_ri = x_ri_bev[:, 0:5, :, :]
        x_bev = x_ri_bev[:, 5:10, :, :]

        feature_ri = self.featureExtracter_RI(x_ri)
        feature_bev = self.featureExtracter_BEV(x_bev)

        feature_ri = feature_ri.squeeze(-1)
        feature_bev = feature_bev.squeeze(-1)
        feature_ri = feature_ri.permute(0, 2, 1)
        feature_bev = feature_bev.permute(0, 2, 1)
        feature_ri = F.normalize(feature_ri, dim=-1)
        feature_bev = F.normalize(feature_bev, dim=-1)

        feature_ri = self.norm_1(feature_ri)
        feature_bev = self.norm_1(feature_bev)

        feature_fuse1 = feature_bev + self.attn1(feature_bev, feature_ri, feature_ri, mask=None)
        feature_fuse1 = self.norm_2(feature_fuse1)
        feature_fuse1 = feature_fuse1 + self.ff1(feature_fuse1)

        feature_fuse2 = feature_ri + self.attn2(feature_ri, feature_bev, feature_bev, mask=None)
        feature_fuse2 = self.norm_3(feature_fuse2)
        feature_fuse2 = feature_fuse2 + self.ff2(feature_fuse2)

        feature_fuse1_ext = feature_fuse1 + self.attn1_ext(feature_fuse1, feature_ri, feature_ri, mask=None)
        feature_fuse1_ext = self.norm_2_ext(feature_fuse1_ext)
        feature_fuse1_ext = feature_fuse1_ext + self.ff1_ext(feature_fuse1_ext)

        feature_fuse2_ext = feature_fuse2 + self.attn2_ext(feature_fuse2, feature_bev, feature_bev, mask=None)
        feature_fuse2_ext = self.norm_3_ext(feature_fuse2_ext)
        feature_fuse2_ext = feature_fuse2_ext + self.ff2_ext(feature_fuse2_ext)

        feature_fuse = torch.cat((feature_fuse1_ext, feature_fuse2_ext), dim=-2)
        feature_cat_origin = torch.cat((feature_bev, feature_ri), dim=-2)
        feature_fuse = torch.cat((feature_fuse, feature_cat_origin), dim=-1)

        feature_fuse = feature_fuse.permute(0, 2, 1)

        feature_com = feature_fuse.unsqueeze(3)

        feature_com = F.normalize(feature_com, dim=1)
        feature_com = self.net_vlad(feature_com)
        feature_com = F.normalize(feature_com, dim=1)

        feature_ri = feature_ri.permute(0, 2, 1)
        feature_ri = feature_ri.unsqueeze(-1)
        feature_ri_enhanced = self.net_vlad_ri(feature_ri)
        feature_ri_enhanced = F.normalize(feature_ri_enhanced, dim=1)

        feature_bev = feature_bev.permute(0, 2, 1)
        feature_bev = feature_bev.unsqueeze(-1)
        feature_bev_enhanced = self.net_vlad_ri(feature_bev)
        feature_bev_enhanced = F.normalize(feature_bev_enhanced, dim=1)
        feature_com = torch.cat((feature_ri_enhanced, feature_com), dim=1)
        feature_com = torch.cat((feature_com, feature_bev_enhanced), dim=1)

        return feature_com

