from torch import nn
import torch
from torch.nn import functional as F
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
# import torch
# import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding


class LSTMEncoder(nn.Module):
    def __init__(self, opt, weights):
        super(LSTMEncoder, self).__init__()
        self.num_support = opt.ways * opt.shots
        self.hidden_dim = opt.hidden_dim

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights,dtype=torch.float),freeze=False)
        # if weights is not None:
        #     self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.bilstm = nn.LSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * opt.hidden_dim, opt.output_dim)
        self.fc2 = nn.Linear(opt.output_dim, opt.output_dim)

    def attention(self, x):
        weights = torch.tanh(self.fc1(x))
        weights = self.fc2(weights)  # (batch=k*c, seq_len, d_a)
        batch, seq_len, d_a = weights.shape
        weights = weights.transpose(1, 2)  # (batch=k*c, d_a, seq_len)
        weights = weights.contiguous().view(-1, seq_len)
        weights = F.softmax(weights, dim=1).view(batch, d_a, seq_len)
        sentence_embeddings = torch.bmm(weights, x)  # (batch=k*c, d_a, 2*hidden)
        avg_sentence_embeddings = torch.mean(sentence_embeddings, dim=1)  # (batch, 2*hidden)
        return avg_sentence_embeddings

    def forward(self, x, hidden=None,num_support=None):
        batch_size, _ = x.shape
        if num_support is None:
            num_support = self.num_support
        if hidden is None:
            h = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h, c = hidden
        x = self.embedding(x)
        outputs, _ = self.bilstm(x, (h, c))  # (batch=k*c,seq_len,2*hidden)
        outputs = self.attention(outputs)  # (batch=k*c, 2*hidden)
        # (c*s, 2*hidden_size), (c*q, 2*hidden_size)
        support, query = outputs[0: num_support], outputs[num_support:]
        return support, query





class ATAE_LSTMEncoder(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTMEncoder, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.ways)

    def forward(self, text_indices,aspect_indices):
        x_len = torch.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        return output

