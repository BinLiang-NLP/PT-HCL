from torch import nn
import torch
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    def __init__(self, opt, weights):
        super(CNNEncoder, self).__init__()
        self.num_support = opt.ways * opt.shots
        self.hidden_dim = opt.hidden_dim

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights,dtype=torch.float),freeze=False)
        # if weights is not None:
        #     self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.conv = nn.Conv1d(opt.embed_dim,2*self.hidden_dim,kernel_size=3,padding=1)
        self.pool = nn.MaxPool1d(opt.max_seq_len)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None,num_support=None):
        batch_size, _ = x.shape
        if num_support is None:
            num_support = self.num_support
        senten_embedding = self.embedding(x)
        shape = list(tuple(senten_embedding.shape))
        seq_length, embedding_dim = shape[-2],shape[-1]
        feature_map = self.conv(senten_embedding.view(-1, seq_length, embedding_dim).transpose(1, 2))
        feature_map = self.relu(feature_map)
        features = self.pool(feature_map)
        ori = tuple(shape[:-2] + [-1])
        outputs = features.view(ori)
        support, query = outputs[0: num_support], outputs[num_support:]
        return support, query

