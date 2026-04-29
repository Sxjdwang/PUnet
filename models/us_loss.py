from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, anchor_video, positive_audio, negative_audio):
        device = anchor_video.device

        anchor_feature = anchor_video  # audio feature
        positive_audio = positive_audio.unsqueeze(0)
        contrast_feature = torch.cat([positive_audio, negative_audio], dim=0)

        # compute logits
        anchor_dot_contrast = torch.sum(anchor_video.unsqueeze(0)*contrast_feature, dim=-1) / self.temperature
        anchor_dot_contrast = anchor_dot_contrast.view(anchor_dot_contrast.shape[0], -1).permute(1, 0)

        mask = torch.zeros_like(anchor_dot_contrast).to(device)
        mask[:, 0] = 1.
        mask = mask.detach()

        loss = self.lossBySimi(anchor_dot_contrast, mask)

        return loss

    def lossBySimi(self, similarity, mask):
        # for numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).to(device),
        #     0
        # )

        # mask = mask * logits_mask

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        sum_log_prob_pos = (mask * log_prob).sum(1)
        mean_log_prob_pos = sum_log_prob_pos #/ (mask.sum(1))   as mask.sum(1) = 1

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss