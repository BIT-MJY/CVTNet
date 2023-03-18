# CVTNet


A Cross-View Transformer Network for Place Recognition Using LiDAR Data. [arXiv](https://arxiv.org/abs/2302.01665)

CVTNet fuses the range image views (RIVs) and bird's eye views (BEVs) generated from LiDAR data to recognize previously visited places.

Developed by [Junyi Ma](https://github.com/BIT-MJY) and [Xieyuanli Chen](https://github.com/Chen-Xieyuanli).

### Table of Contents
1. [Publications](#Publications)
2. [Dependencies](#Dependencies)
3. [How to use](#How-to-use)
4. [Data preparation](#Data-preparation)
5. [License](#License)

## Publications

If you use the code in your work, please cite our [paper](https://ieeexplore.ieee.org/document/9994714):

```
@misc{ma2023cvtnet,
      title={CVTNet: A Cross-View Transformer Network for Place Recognition Using LiDAR Data}, 
      author={Junyi Ma and Guangming Xiong and Jingyi Xu and Xieyuanli Chen},
      year={2023},
      eprint={2302.01665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dependencies

Please refer to our [SeqOT repo](https://github.com/BIT-MJY/SeqOT).

## How to use

We provide a training and test tutorial for NCLT sequences in this repository. 

### Training

You can start the training process with

```
cd train
python ./train_cvtnet.py
```
Note that we only train our model using the oldest sequence of NCLT dataset (2012-01-08), to prove that our model works well for long time spans even if seeing limited data.  

### Test

You can test CVTNet by

```
cd test
python ./test_cvtnet_prepare.py
python ./cal_topn_recall.py
```

## License

Copyright 2023, Junyi Ma, Guangming Xiong, Jingyi Xu, Xieyuanli Chen, Beijing Institute of Technology.

This project is free software made available under the MIT License. For more details see the LICENSE file.
