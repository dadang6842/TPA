import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=None, g=1):
        super().__init__()
        self.c = nn.Conv1d(c_in, c_out, k, s, k//2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.c(x)))

class MultiPathCNN(nn.Module):
    """CNN Backbone: Input [B, T, C] -> Output [B, T, D]. Handles internal C/T transpose."""
    def __init__(self, in_ch=9, d_model=128, branches=(3,5,9,15), stride=2):
        super().__init__()
        h = d_model // 2
        self.pre = ConvBNAct(in_ch, h, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(ConvBNAct(h, h, k, stride, g=h), ConvBNAct(h, h, 1))
            for k in branches
        ])
        self.post = ConvBNAct(len(branches)*h, d_model, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        fmap = self.post(torch.cat([b(self.pre(x)) for b in self.branches], dim=1))
        return fmap.transpose(1, 2)