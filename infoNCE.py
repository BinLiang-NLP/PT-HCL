from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1)
        features = F.normalize(features, dim=1)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # contrast_count = 2  # 16 ，768

        # print(features.shape)

        # features = features.unsqueeze(dim=1)
        # features = F.normalize(features, dim=1)
        # features = features.repeat(contrast_count, 1, 1).squeeze(1)
        # contrast_feature = features
        #-----



        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask

        # compute logits
        # similarity = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        #-----
        logits = torch.mm(anchor_feature, contrast_feature.t()) / self.temperature
        logits_min, _ = torch.min(logits, dim=1, keepdim=True)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        _range = logits_max - logits_min
        logits = torch.div(logits - logits_min, _range)
        #-----


        similarity = torch.exp(logits)
        # print(similarity)



        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg+1))))




        return loss