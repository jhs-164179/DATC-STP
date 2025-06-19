import os
import random
import argparse
import numpy as np
import torch


def check_path(root):
    if not os.path.exists(root):
        os.makedirs(root)


def seed_everything(seed:int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

        
def parse_tuple(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in the format (x, y)")


def print_f(contents, file):
    print(contents)
    print(contents, file=file)        


def makeparser():
    parser = argparse.ArgumentParser()
    # =====Train params=====
    parser.add_argument('--dataset', type=str, default='kth20', help='Dataset name')
    parser.add_argument('--pred_len', type=int, default=20, help='Only for testing long-term prediction with KTH')
    parser.add_argument('--in_shape', type=parse_tuple, default=(10, 1, 128, 128), help='Plz provide as "(T,C,H,W)"')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--sched', type=str, default='onecycle', help='Scheduler')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--loss', type=str, default='l1l2', help='Loss function')
    parser.add_argument('--batch_size', type=int, default=8, help='Train batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
    parser.add_argument('--save_path', type=str, default='kth', help='Experiment name')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    # =====Model params=====
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension2 for deep patch embedding")
    parser.add_argument('--patch_size', type=int, default=2, help="Patch size for patch embedding")
    parser.add_argument('--N', type=int, default=6, help="Number of blocks")
    parser.add_argument('--S_kernel', type=int, default=7, help="Spatio block kernel size")
    parser.add_argument('--droppath', type=float, default=0.0, help='Droppath rate')
    parser.add_argument('--ratio', type=int, default=2, help='MLP ratio')    
    # =====Test(Inference) params=====
    parser.add_argument('--save_result', action='store_true', help='Save or not the inference results and groud-truth')
    parser.add_argument('--pr_path', type=str, default='pred_kth.npy', help='Set the directory for saving the inference results')
    parser.add_argument('--gt_path', type=str, default='gt_kth.npy', help='Set the directory for saving the ground-truth results')
    args = parser.parse_args()
    return args