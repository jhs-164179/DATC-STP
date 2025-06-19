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

