import torch
from torch import nn
from torch.nn import functional as F


def squash(tensor):
    norm = (tensor * tensor).sum(-1)
    scale = norm / (1+norm)
    return scale.unsqueeze(-1) * tensor / torch.sqrt(norm).unsqueeze(-1)


class Induction(nn.Module):
    def __init__(self, opt):
        super(Induction, self).__init__()
        self.C = opt.ways
        self.S = opt.shots
        self.H = opt.feature_dim
        self.iterations = opt.iterations
        self.W = torch.nn.Parameter(torch.randn(self.H, self.H))

    def forward(self, x):
        self.S = x.shape[0] // self.C
        b_ij = torch.zeros(self.C, self.S).to(x)
        # x = squash(x)
        for _ in range(self.iterations):
            d_i = F.softmax(b_ij.unsqueeze(2), dim=1)  # (C,S,1)
            e_ij = torch.mm(x.reshape(-1, self.H), self.W).reshape(self.C, self.S, self.H)  # (C,S,H)
            c_i = torch.sum(d_i * e_ij, dim=1)  # (C,H)
            # squash
            squared = torch.sum(c_i ** 2, dim=1).reshape(self.C, -1)
            coeff = squared / (1 + squared) / torch.sqrt(squared + 1e-9)
            c_i = coeff * c_i
            c_produce_e = torch.bmm(e_ij, c_i.unsqueeze(2))  # (C,S,1)
            b_ij = b_ij + c_produce_e.squeeze(2)
        return c_i

