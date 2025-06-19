from torch import nn
from module import *


class DATCSTP(nn.Module):
    def __init__(self, c_in, embed_dim, patch_size, N, kernel_size, T, ratio=2, drop_path=0.2):
        super().__init__()
        new_T = T // patch_size
        self.pe = PatchEmbed_CNN3D(c_in, embed_dim, patch_size)
        net = []
        for _ in range(N):
            net.append(Block(embed_dim, kernel_size=kernel_size, T=new_T, ratio=ratio, drop_path=drop_path))
        self.net = nn.ModuleList(net)
        self.pb = PatchEmbed_CNN3D(c_in, embed_dim, patch_size, patch_back=True)

    def forward(self, x):
        x = self.pe(x)
        for net in self.net:
            x = net(x)
        x = self.pb(x)
        return x