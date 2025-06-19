import random
import torch
from torch.utils.data import Dataset


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y, use_augment=False, data_name='taxibj'):
        super(TaxibjDataset, self).__init__()
        self.X = (X+1) / 2  # channel is 2
        self.Y = (Y+1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels