import torch
from torch import nn
from torch.nn import functional as F


class Relation(nn.Module):
    def __init__(self, opt):
        super(Relation, self).__init__()
        self.C = opt.ways
        self.out_size = opt.relation_dim
        self.H = opt.feature_dim
        self.M = torch.nn.Parameter(torch.randn(self.H, self.H, self.out_size))
        self.W = torch.nn.Parameter(torch.randn(self.C * self.out_size, self.C))
        self.b = torch.nn.Parameter(torch.randn(self.C))

    def forward(self, class_vector, query_encoder):  # (C,H) (Q,H)
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,Q)
        V = F.relu(mid_pro.transpose(0, 1))  # (Q,C*out_size)
        probs = torch.sigmoid(torch.mm(V, self.W) + self.b)  # (Q,C)
        return probs