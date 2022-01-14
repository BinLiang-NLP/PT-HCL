from torch import nn
import torch


class RelationCompare(nn.Module):
    def __init__(self,ways,feature_dim):
        super(RelationCompare, self).__init__()
        self.ways = ways
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(nn.Linear(self.feature_dim*2,64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64,1),
                                nn.Sigmoid())

    def forward(self, class_vector,query_encoder): #(ways,feat_dim) (query_size,feat_dim)
        query_size = query_encoder.shape[0]
        expanded_class_vector = class_vector.unsqueeze(0).expand(query_size,-1,-1) ## querysize * ways * feature_dim
        expanded_query_encoder = query_encoder.unsqueeze(1).expand(-1,self.ways,-1) ## querysize * ways * feature_dim
        concatenated_vector = torch.cat((expanded_query_encoder,expanded_class_vector),dim=2) ##querysize * ways* (2*feature_dim)
        scores = self.fc(concatenated_vector).squeeze(dim=-1) # (query_size, ways)
        return scores