# CVTNet


A Cross-View Transformer Network for Place Recognition Using LiDAR Data. [[arXiv](https://arxiv.org/abs/2302.01665)]

[Junyi Ma](https://github.com/BIT-MJY), Guangming Xiong, [Jingyi Xu](https://github.com/BIT-XJY), [Xieyuanli Chen*](https://github.com/Chen-Xieyuanli)

<img src="https://github.com/BIT-MJY/CVTNet/blob/main/motivation.png" width="70%"/>

CVTNet fuses the range image views (RIVs) and bird's eye views (BEVs) generated from LiDAR data to recognize previously visited places. RIVs and BEVs have the same shift for each yaw-angle rotation, which can be used to extract aligned features.

<img src="https://github.com/BIT-MJY/CVTNet/blob/main/corresponding_rotation.gif" >  

### Table of Contents
1. [Publication](#Publication)
2. [Dependencies](#Dependencies)
3. [How to use](#How-to-use)
4. [TODO](#Miscs)
5. [Miscs](#Miscs)
6. [License](#License)

## Publication

If you use the code in your work, please cite our [paper](https://arxiv.org/abs/2302.01665):

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

We provide a training and test tutorial for NCLT sequences in this repository. Before any operation, please modify the [config file](https://github.com/BIT-MJY/CVTNet/tree/main/config) according to your setups.

### Data preparation

* laser scans from NCLT dataset: [[2012-01-08](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-01-08_vel.tar.gz)]       [[2012-02-05](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-02-05_vel.tar.gz)]
* [pretrained model](https://drive.google.com/file/d/1iQEY-DMDxchQ2RjG4RPkQcQehb_fujvO/view?usp=share_link)
* [training indexes](https://drive.google.com/file/d/1jEcnuHjEi0wqe8GAoh6UTa4UXTu0sDPr/view)
* [ground truth](https://drive.google.com/file/d/13-tpLQiHK4krd-womDV6UPevvHUIFNyF/view?usp=share_link)

You need to generate RIVs and BEVs from raw LiDAR data by

```
cd tools
python ./gen_ri_bev.py
```

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

### C++ implementation

We provide a toy example showing C++ implementation of CVTNet with libtorch. First, you need to generate the model file by

```
cd CVTNet_libtorch
python ./gen_libtorch_model.py
```

* Before building, make sure that [PCL](https://github.com/PointCloudLibrary/pcl) exists in your environment.
* Here we use [LibTorch for CUDA 11.3 (Pre-cxx11 ABI)](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip). Please modify the path of **Torch_DIR** in [CMakeLists.txt](https://github.com/BIT-MJY/CVTNet/blob/main/CVTNet_libtorch/ws/CMakeLists.txt). 
* For more details of LibTorch installation, please check this [website](https://pytorch.org/get-started/locally/).  

Then you can generate a descriptor of the provided 1.pcd by

```
cd ws
mkdir build
cd build
cmake ..
make -j6
./fast_cvtnet
```

## TODO
- [x] Release the preprocessing code and pretrained model
- [ ] Release sequence-enhanced CVTNet (SeqCVT)

## Miscs

Thanks for your interest in our previous work for LiDAR-based place recognition.

* [OverlapNet](https://github.com/PRBonn/OverlapNet): Loop Closing for 3D LiDAR-based SLAM
* [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer): An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition
* [SeqOT](https://github.com/BIT-MJY/SeqOT): A Spatial-Temporal Transformer Network for Place Recognition Using Sequential LiDAR Data

## License

Copyright 2023, Junyi Ma, Guangming Xiong, Jingyi Xu, Xieyuanli Chen, Beijing Institute of Technology.

This project is free software made available under the MIT License. For more details see the LICENSE file.
