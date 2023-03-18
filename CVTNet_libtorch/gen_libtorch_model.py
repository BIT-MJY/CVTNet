# Developed by Junyi Ma and Xieyuanli Chen
# This file is covered by the LICENSE file in the root of the project CVTNet:
# https://github.com/BIT-MJY/CVTNet
# Brief: generate .pt for C++ implementation

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import torch
import yaml
from modules.cvtnet import CVTNet

config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
pretrained_weights = config["cpp_implementation"]["weights"]
checkpoint = torch.load(pretrained_weights)
amodel = CVTNet(channels=5, use_transformer=True)
amodel.load_state_dict(checkpoint['state_dict'])
amodel.cuda()
amodel.eval()
example = torch.rand(1, 10, 32, 900)
example = example.cuda()
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(amodel, example)
traced_script_module.save("./CVTNet.pt")


