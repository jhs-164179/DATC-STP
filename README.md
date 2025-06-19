# DATC-STP: Towards Accurate yet Efficient Spatiotemporal Prediction with Transformer-style CNN
![Image](https://github.com/user-attachments/assets/96c79df2-40f4-48be-a32c-69f311ecb078)<br>
An implementation of "DATC-STP: Towards Accurate yet Efficient Spatiotemporal Prediction with Transformer-style CNN." IEEE ACCESS (2025). **[[paper](https://doi.org/10.1109/ACCESS.2025.3573639)]**<br>

## Abstract
Recently, convolutional neural networks (CNNs) or vision transformers (ViTs) based Multi-In-Multi-Out (MIMO) architectures are proposed to overcome the limitations of recurrent neural networks (RNNs) based Single-In-Single-Out (SISO) architectures. These architectures prevent the inherent limitations of RNNs, which degrade performance and inefficiency of parallelization due to the sequential properties. However, there are still some challenges. CNN-based MIMO architectures have difficulty capturing global spatiotemporal information due to the local properties of its kernel. Meanwhile, ViT-based MIMO architectures have difficulty capturing local spatiotemporal information and require high-computational resource due to the self-attention. To improve MIMO architecture with overcome these limitations, we propose a novel accurate yet efficient Dual-Attention Transformer-style CNN for Spatiotemporal Prediction (DATC-STP).DATC-STP captures both local and global spatiotemporal information by 3D patch embedding and Transformer-style CNN. Specifically, 3D patch embedding extract local spatiotemporal features and reduce the size of input data including temporal, height, and width. Two Transformer-style CNN based attention blocks treat spatiotemporal data similarly with image and capture global information with CNNs. These structure makes DATC-STP accurate yet efficient. To demonstrate the effectiveness of DATC-STP, we conduct comprehensive experiments with three promising benchmark datasets, MovingMNIST, TaxiBJ, and KTH.We evaluated that the proposed DATC-STP achieves both competitive performance and efficient. Furthermore, results of ablation study demonstrates the useful for each component of DATC-STP and highlights the potential of proposed methods.

## Installation
```shell
# for fast start
pip install -r requirements.txt
```

### requirements
- torch
- fvcore
- timm == 1.0.12
- scikit-image == 0.19.3

## Structures

- `dataset/` Where code for torch dataset, and downloaded data will be stored. All the datasets have to be located like this (e.g. dataset/moving_mnist/~.npy | dataset/taxibj/~.npy | dataset/kth/[action_name]/~.jpg)
- `logs/` Where txt file which logged train process of each dataset are contained. When evaluating with test.py performed, testing log and saved inference will be saved in here.
- `model.py` Define DATC-STP.
- `module.py` Define DATC-STP modules include 3D P.E and P.B, Two attention blocks, etc.
- `metrics.py` Define the metrics code for evaluating (MSE, MAE, PSNR, SSIM).
- `utils.py` Define the code for utilization including argparse, etc.
- `main.py` Training process of DATC-STP is contained.
- `test.py` Testing process of DATC-STP is contained (only for evaluating).
- `*.sh` Train/Test bash code for each dataset.

## Dataset description
### MovingMNIST
- Download [MNIST](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy).
### TaxiBJ
- Download [TaxiBJ dataset](https://github.com/TolicWang/DeepST/blob/master/data/TaxiBJ/README.md) or we can use [OpenSTL](https://github.com/chengtan9907/OpenSTL).
### KTH
- Download by using [OpenSTL](https://github.com/chengtan9907/OpenSTL).

## Train

```shell
# after remove comment in first paragraph
# for train moving_mnist dataset
bash test_moving_mnist.sh
# for train taxibj dataset
bash test_taxibj.sh
# for train kth dataset
bash test_kth.sh
```

## Test

```shell
# without remove comment
# for test moving_mnist dataset with saved params
bash test_moving_mnist.sh
# for test taxibj dataset with saved params
bash test_taxibj.sh
# for test kth dataset with saved params
bash test_kth.sh
```

## Citation

```
@ARTICLE{11015442,
  author={Jin, Hyeonseok and Kim, Kyungbaek},
  journal={IEEE Access}, 
  title={DATC-STP: Towards Accurate yet Efficient Spatiotemporal Prediction with Transformer-style CNN}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Spatiotemporal phenomena;Computer architecture;Accuracy;Transformers;Feature extraction;Convolutional neural networks;Correlation;Long short term memory;Three-dimensional displays;Kernel;Spatiotemporal prediction;multi-in-multi-out;convolutional neural network;vision transformer;deep learning},
  doi={10.1109/ACCESS.2025.3573639}}
```
