"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import math
import torch
from torch.utils.data import Dataset
from .preprocess import AddNoise
from .utilsstft import prepare_main_input
from .utilsstft import stft_tran, collate_fn
import numpy as np
import random


class LRS2Main(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, audiodir, videodir, reqInpLen, charToIx, stepSize, audioParams, videoParams, noiseParams, wholeset, process_conf=None, e2etrain={'audio': False, 'video': True}, rank=0, nshard=1, ap=None):
        super(LRS2Main, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        nlength = math.ceil(len(lines) / nshard)
        start_id, end_id = nlength * rank, nlength * (rank + 1)
        lines = lines[start_id: end_id]

        self.txtlist = [datadir + "/main/" + line.strip().split(' ')[0] for line in lines]
        self.datalist = [audiodir + "/main/" + line.strip().split(' ')[0] for line in lines]
        self.videolist = [videodir + "/main/" + line.strip().split(' ')[0] for line in lines]
        self.audiosuffix = '.wav'

        self.preprocess = prepare_main_input
        self.videosuffix = '.mp4'

        self.reqInpLen = reqInpLen
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        self.videoParams = videoParams

        self.noiseGenerator = stft_tran(audioParams["stftWindow"], audioParams["stftWinLen"], audioParams["stftOverlap"],
                                    noiseParams['FILE'], noiseParams['LEVEL'], noiseParams['Prob'], process_conf)
        return


    def __getitem__(self, index):
        #using the same procedure as in pretrain dataset class only for the train dataset
        if "train" in self.dataset:
            base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = np.random.choice(ixs)

        #passing the visual features file and the target file paths to the prepare function to obtain the input tensors
        audioFeaturesFile = self.datalist[index] + self.audiosuffix
        targetFile = self.txtlist[index] + ".txt"
        videoFeatureFile = self.videolist[index] + self.videosuffix
        inp, trgt, inpLen, reqLen = self.preprocess(videoFeatureFile, audioFeaturesFile, targetFile, self.reqInpLen, self.charToIx, self.videoParams, self.noiseGenerator, 'train' in self.dataset)#
        return inp, trgt, inpLen, reqLen


    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        if "train" in self.dataset:
            return self.stepSize
        else:
            return len(self.datalist)


class avsrelement(object):
    def __init__(self, video, audio, text, frame, id):
        super(avsrelement, self).__init__()
        self.video = video
        self.text = text
        self.frame = frame
        self.id = id
        self.audio = audio


class LRS2MixGroup(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    Supposed to be involved to E2E training with mixtrain set
    """
    def __init__(self, dataset, datadir, audiodir, videodir, batchsize, reqInpLen, charToIx, stepSize, audioParams, videoParams,
                 noiseParams, max_len=600, process_conf=False, short_startup=True):
        super(LRS2MixGroup, self).__init__()

        max_len_in = 150
        self.shortfirst = short_startup
        with open(datadir + "/" + dataset + "_length.txt", "r") as f:
            lines = f.readlines()
        fulllist = []
        for i, line in enumerate(lines):
            info = line.strip().split(' ')
            fulllist.append(avsrelement(videodir + "/" + info[0],
                                        audiodir + "/" + info[0],
                                        datadir + "/" + info[0],
                                        int(info[1]), i))

        self.stepSize = int(stepSize)
        self.sortedlist, self.boundary, self.bz = self.sort_group(fulllist, batchsize, max_len, max_len_in, True)
        print(len(self.sortedlist), self.boundary, self.bz)
        self.reformbatch()

        self.audiosuffix = '.wav'

        self.preprocess = prepare_main_input
        self.videosuffix = '.mp4'

        self.reqInpLen = reqInpLen
        self.charToIx = charToIx
        self.dataset = dataset
        self.audioParams = audioParams
        self.videoParams = videoParams

        self.noiseGenerator = stft_tran(audioParams["stftWindow"], audioParams["stftWinLen"], audioParams["stftOverlap"],
                                        noiseParams['FILE'], noiseParams['LEVEL'], noiseParams['Prob'], process_conf)
        return

    def choose_section(self):
        self.roundprob[self.section] = 0
        if np.sum(self.roundprob) != 0:
            self.roundprob = self.roundprob / np.sum(self.roundprob)
            self.section = int(np.random.choice(len(self.roundprob), 1, replace=False, p=self.roundprob))

    def reformbatch(self):
        start = 0
        batched = []
        for i, boundary in enumerate(self.boundary):
            candidate = list(range(start, boundary))
            random.shuffle(candidate)
            batched += self.takesample(candidate, self.bz[i])
            start = boundary
            if self.shortfirst:
                break
        round = int(np.ceil(len(batched)/self.stepSize))
        print(len(batched), self.stepSize, round)
        compliment = int(round * self.stepSize - len(batched))
        full_id = list(range(len(batched)))
        random.shuffle(full_id)
        batched += [batched[j] for j in full_id[:compliment]]
        self.datalist, self.roundprob = batched, np.ones(round)/round
        self.section = int(np.random.choice(len(self.roundprob), 1, replace=False, p=self.roundprob))
        print(len(self.datalist))

    def takesample(self, id_list, bs):
        len_list = len(id_list)
        batch_group = []
        for i in range(0, len_list, bs):
            stop = min(len_list, i+bs)
            ids = id_list[i: stop]
            minibatch = [self.sortedlist[j] for j in ids]
            batch_group.append(minibatch)
            if stop == len_list:
                break
        return batch_group

    def __getitem__(self, index):
        #using the same procedure as in pretrain dataset class only for the train dataset
        if "train" in self.dataset:
            # section = int(np.random.choice(len(self.roundprob), 1, replace=False, p=self.roundprob))
            index = self.section * self.stepSize + index

        group = self.datalist[int(index)]
        group_retrieve = []
        for sample in group:
            audioFeaturesFile = sample.audio + self.audiosuffix
            targetFile = sample.text + ".txt"
            videoFeatureFile = sample.video + self.videosuffix
            group_retrieve.append(self.preprocess(videoFeatureFile, audioFeaturesFile, targetFile, self.reqInpLen,
                                                  self.charToIx, self.videoParams, self.noiseGenerator, 'train' in self.dataset))
        return collate_fn(group_retrieve)

    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        if "train" in self.dataset:
            return self.stepSize
        else:
            return len(self.datalist)
        # return len(self.datalist)

    def sort_group(self, fulllist, batch_size, max_len=600, next_boundary=150, group=False):
        boundary = []
        batch_sizes = []
        sorted_data = sorted(
            fulllist,
            key=lambda data: int(data.frame),
            reverse=False,
        )
        if group:
            len_sort = len(sorted_data)
            for i in range(1, len(sorted_data)):
                if sorted_data[len_sort - i].frame < max_len:
                    break
            if i > 1:
                del sorted_data[-i:]
            for i, data in enumerate(sorted_data):
                if data.frame > next_boundary:
                    boundary.append(i)
                    batch_sizes.append(int(batch_size))
                    batch_size /= 2
                    next_boundary *= 2
                    if next_boundary == max_len:
                        boundary.append(len(sorted_data))
                        batch_sizes.append(int(batch_size))
                        break
        else:
            boundary.append(len(sorted_data))
            batch_sizes.append(batch_size)
        return sorted_data, boundary, batch_sizes
