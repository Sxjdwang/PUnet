"""
Employ l2 normalization methods
three layers or four layers
"""
import torch
import numpy as np
import sys
import scipy.io as io
import torch.nn as nn
import torch.nn.functional as F

from models.us_loss import SupConLoss
from typing import Optional, Tuple


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape


    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    Output:
        mask: its shape is same with padding_mask where the value of masked frame is True
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    mask_idcs = []
    for i in range(bsz):

        # actual timesteps
        sz = all_sz - padding_mask[i].long().sum().item()
        if sz < 25:
            mask_idc = np.arange(0, sz)
        else:
            # number of indices of starting mask
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)

            lengths = np.full(num_mask, mask_length)

            if sum(lengths) == 0:
                lengths[0] = min(mask_length, sz - 1)

            """Random sample (overlap allowed)"""
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            # all indices of masked timesteps
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
        """Random sample (overlap allowed)"""

    min_len = min([len(m) for m in mask_idcs])
    # make numbers of masked timesteps of all sentence equal
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def sample_negatives(y, n_negatives=15, cross_sample_negatives=85):

    bsz, tsz, fsz = y.shape
    num = tsz

    # FIXME: what happens if padding_count is specified?
    cross_high = tsz * bsz
    high = tsz
    with torch.no_grad():
        assert high > 1, f"{bsz, tsz, fsz}"

        index_range = torch.arange(num).unsqueeze(-1)

        if n_negatives > 0:
            tszs = (index_range
                    .expand(-1, n_negatives)
                    .flatten()
            )
            """Problem: repeated indices exist"""
            # neg_idxs = torch.randint(
            #     low=0, high=high - 1, size=(bsz, n_negatives * num)
            # )
            """Fix the mentioned problem"""
            neg_idxs = torch.multinomial(torch.ones((bsz*num, high-1), dtype=torch.float), n_negatives)\
                .view(bsz, n_negatives * num)

            neg_idxs[neg_idxs >= tszs] += 1

        replacement = False if bsz*tsz > (100-n_negatives) else True

        if cross_sample_negatives > 0:
            tszs = (index_range
                    .expand(-1, cross_sample_negatives)
                    .flatten()
            )
            """Problem: repeated indices exist"""
            # cross_neg_idxs = torch.randint(
            #     low=0,
            #     high=cross_high - 1,
            #     size=(bsz, cross_sample_negatives * num),
            # )
            """Fix the mentioned problem"""
            cross_neg_idxs = torch.multinomial(torch.ones((bsz*num, cross_high-1), dtype=torch.float), cross_sample_negatives, replacement=replacement)\
                .view(bsz, cross_sample_negatives * num)

            cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if n_negatives > 0:
        for i in range(1, bsz):
            neg_idxs[i] += i * high
    else:
        neg_idxs = cross_neg_idxs

    """Problem: negatives of former frames are all within utterance while that of latter frames are almost from other utterances"""
    # if cross_sample_negatives > 0 and n_negatives > 0:
    #     neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
    """Fix the mentioned problem"""
    if cross_sample_negatives > 0 and n_negatives > 0:
        neg_idxs = neg_idxs.view(bsz, num, n_negatives)
        cross_neg_idxs = cross_neg_idxs.view(bsz, num, cross_sample_negatives)
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=2)
        neg_idxs = neg_idxs.view(bsz, -1)
    #!Todo: neg_idxs and cross_neg_idxs still have same indices

    return neg_idxs


class contrastive_loss(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(contrastive_loss, self).__init__()

        self.audionorm = nn.Linear(in_channel, out_channel)

        self.videonorm = nn.Linear(in_channel, out_channel)

        self.criterion = SupConLoss()

    def forward(self, video, audio, padding_mask):
        """

        :param video: tensor-> batch*ndim*nframe, output of transformer
        :param audio: tensor-> batch*ndim*nframe, output of transformer
        :param padding_mask: tensor-> batch*nframe
        :return:
        """

        video = F.normalize(video, dim=2)
        video_cl = self.videonorm(video)
        video_cl_norm = F.normalize(video_cl, dim=2)

        audio = F.normalize(audio, dim=2)
        audio_cl = self.audionorm(audio)
        audio_cl_norm = F.normalize(audio_cl, dim=2)

        mask = compute_mask_indices((video.shape[0], video.shape[1]), padding_mask, 0.5, 4, 2)
        torch_mask = torch.from_numpy(mask).to(audio_cl_norm.device)

        fsz = audio_cl_norm.shape[2]

        feature_v = video_cl_norm[torch_mask].view(audio_cl_norm.shape[0], -1, fsz)
        feature_a = audio_cl_norm[torch_mask].view(audio_cl_norm.shape[0], -1, fsz)
        # print(feature_a.shape[1])
        # print(torch.sum((~padding_mask).float(), dim=1))

        # n_negatives = min(50, int(feature_v.shape[1]*2/3))
        n_negatives = min(80, int(feature_v.shape[1]*2/3))
        neg_idxs = sample_negatives(feature_v, n_negatives=n_negatives, cross_sample_negatives=100-n_negatives)

        #!Todo: May the positive occur in the negatives?
        negs_a = feature_a.view(-1, fsz)[neg_idxs.view(-1)]
        negs_a = negs_a.view(
            feature_a.shape[0], feature_a.shape[1], 100, fsz
        ).permute(2, 0, 1, 3)  # to NxBxTxC

        dist_loss = self.criterion(feature_v, feature_a, negs_a)

        return dist_loss


