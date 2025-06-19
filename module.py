import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath


class PatchEmbed_CNN3D(nn.Module):
    def __init__(self, c_in, embed_dim, patch_size, patch_back=False):
        super().__init__()
        new_patch_size = patch_size // 2
        self.patch_back = patch_back
        if patch_back:
            self.net = nn.Sequential(
                nn.ConvTranspose3d(embed_dim, embed_dim//2, kernel_size=new_patch_size, stride=new_patch_size, bias=False),
                nn.GELU(),
                nn.BatchNorm3d(embed_dim//2),
                nn.Conv3d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GELU(),
                nn.BatchNorm3d(embed_dim//2),
                nn.ConvTranspose3d(embed_dim//2, c_in, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv3d(c_in, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU(),
                nn.BatchNorm3d(embed_dim//2),
                nn.Conv3d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GELU(),
                nn.BatchNorm3d(embed_dim//2),
                nn.Conv3d(embed_dim//2, embed_dim, kernel_size=new_patch_size, stride=new_patch_size, bias=False)
            )

    def forward(self, x):
        # B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # -> B, C, T, H, W
        x = self.net(x)
        x = x.permute(0, 2, 1, 3, 4) # -> B, T, C, H, W
        return x
    

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x 
        

class Attention(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.fc1 = nn.Conv2d(dim, dim*ratio, 1)
        self.pos = nn.Conv2d(dim*ratio, dim*ratio, 3, padding=1, groups=dim*ratio)
        self.fc2 = nn.Conv2d(dim *ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, kernel_size, T, ratio, drop_path=0.0):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.satt = Attention(dim, kernel_size)
        self.tatt = Attention(dim*T, kernel_size)
        self.mlp = Mlp(dim, ratio)
        init_value = 1e-6
        self.scale1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True
        )
        self.scale2 = nn.Parameter(
            init_value * torch.ones((dim*T)), requires_grad=True
        )
        self.scale3 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # spatial
        # x = x.view(B*T, C, H, W)
        x = x.reshape(B*T, C, H, W)
        x = x + self.drop_path(self.scale1.unsqueeze(-1).unsqueeze(-1) * self.satt(x))
        x = x.view(B, T, C, H, W)

        # temporal
        # x = x.view(B, T*C, H, W)
        x = x.reshape(B, T*C, H, W)
        x = x + self.drop_path(self.scale2.unsqueeze(-1).unsqueeze(-1) * self.tatt(x))
        x = x.view(B, T, C, H, W)

        # feedforward
        x = x.view(B*T, C, H, W)
        x = x + self.drop_path(self.scale3.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        x = x.view(B, T, C, H, W)
        return x
