"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import os
import torch
import cv2 as cv
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from scipy.special import softmax
from torchaudio import sox_effects as se
from torchaudio.compliance.kaldi import fbank
from torchvision import transforms
from espnet.transform.transformation import Transformation

from scipy.io import wavfile
from scipy import signal
import logging


class specaug(object):
    def __init__(self, time_drop_length, time_drop_block, time_drop_min, fre_drop_length, fre_drop_block):
        self.time_drop_length = time_drop_length
        self.time_drop_block = time_drop_block
        self.time_drop_min = time_drop_min
        self.fre_drop_length = fre_drop_length
        self.fre_drop_block = fre_drop_block

    def augstft(self, feat, sampFre, frame_length):
        length = feat.shape[0] * frame_length
        if length > self.time_drop_min / 2:
            if length < self.time_drop_min:
                time_drop_block = int(np.round(self.time_drop_block / 2))
            else:
                time_drop_block = self.time_drop_block
            for i in range(time_drop_block):
                time = int(random.uniform(0, self.time_drop_length) / frame_length)
                start = int(random.random() * (feat.shape[0] - self.time_drop_length / frame_length))
                feat[start: start + time] = feat.mean()
        fre_interval = np.round(0.5*sampFre/(feat.shape[1]-1))
        num_fre = self.fre_drop_length / fre_interval
        for i in range(self.fre_drop_block):
            fre = int(random.randint(0, num_fre))
            start = int(random.randint(0, feat.shape[1]-num_fre))
            # feat[start: start+fre] = 0
            feat[start: start+fre] = feat.mean()
            # print(start, fre)
        return feat


class stft_tran(object):
    def __init__(self, stftWindow, stftWinLen, stftOverlap, noise, snr, prob, process_conf=None):
        self.stftWindow = stftWindow
        self.stftWinLen = stftWinLen
        self.stftOverlap = stftOverlap
        _, self.noise = wavfile.read(noise)
        self.snr = snr
        self.prob = prob
        self.specaug = specaug(0.4, 2, 4, 1500, 2)
        if process_conf is not None:
            self.preprocessing = Transformation(process_conf)

    def extract(self, inputAudio, sampFreq, istrain):
        """pad the audio to get atleast 4 STFT vectors"""
        if len(inputAudio) < sampFreq * (self.stftWinLen + 3 * (self.stftWinLen - self.stftOverlap)):
            padding = int(np.ceil((sampFreq * (self.stftWinLen + 3 * (self.stftWinLen - self.stftOverlap)) - len(inputAudio)) / 2))
            inputAudio = np.pad(inputAudio, padding, "constant")
        inputAudio = inputAudio / np.max(np.abs(inputAudio))

        """adding noise to the audio"""
        snr = self.snr[random.randint(0, len(self.snr)-1)]
        # snr = 0
        if snr < 9999:#1:# istrain and
            pos = np.random.randint(0, len(self.noise) - len(inputAudio) + 1)
            noise = self.noise[pos:pos + len(inputAudio)]
            noise = noise / np.max(np.abs(noise))
            gain = 10 ** (snr / 10)
            noise = noise * np.sqrt(np.sum(inputAudio ** 2) / (gain * np.sum(noise ** 2)))
            inputAudio = inputAudio + noise

        """normalising the audio to unit power"""
        inputAudio = inputAudio / np.sqrt(np.sum(inputAudio ** 2) / len(inputAudio)) #* 100

        """computing the STFT and taking only the magnitude of it"""
        _, _, stftVals = signal.stft(inputAudio, sampFreq, window=self.stftWindow, nperseg=sampFreq * self.stftWinLen,
                                     noverlap=sampFreq * self.stftOverlap,
                                     boundary=None, padded=False)
        inp = np.abs(stftVals)
        inp = inp.T

        inp = self.preprocessing(inp, None, **{'train': istrain})

        return inp


def framebeforeConv(frame):
    Target_frame = ((frame - 1) * 2 + 2) * 2 + 3
    # Target_frame = 4 * frame
    return Target_frame

def frameafterConv(frame):
    frameafter = ((frame-3) // 2 - 2) // 2 + 1
    # frameafter = frame // 4
    return frameafter


def prepare_main_input(visualFile, audioFile, targetFile, reqInpLen, charToIx, videoParams, audioProcessor, istrain):

    """
    Function to convert the data sample (visual features file, target file) in the main dataset into appropriate tensors.
    """
    if targetFile is not None:

        #reading the target from the target file and converting each character to its corresponding index
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:].replace('{LG} ', '').replace(' {LG}', '').replace('{NS} ', '').replace(' {NS}', '')

        trgt = [charToIx[char] for char in trgt]
        trgt = np.array(trgt)

    sampFreq, inputAudio = wavfile.read(audioFile)

    inpaudio = audioProcessor.extract(inputAudio, sampFreq, istrain)

    captureObj = cv.VideoCapture(visualFile)
    roiSize = 122
    center = 112
    roiSequence = list()
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = cv.resize(grayed, (224, 224))
            roi = grayed[int(center - (roiSize / 2)):int(center + (roiSize / 2)),
                  int(center - (roiSize / 2)):int(center + (roiSize / 2))]
            roiSequence.append(roi)
            # roiSequence.append(frame)
        else:
            break
    captureObj.release()
    inpvideo = np.stack(roiSequence)


    inpLenaudio = frameafterConv(len(inpaudio))
    inpLenvideo = len(inpvideo)

    if inpLenvideo < inpLenaudio:
        leftPadding = int(np.floor((inpLenaudio - inpLenvideo) / 2))
        rightPadding = int(np.ceil((inpLenaudio - inpLenvideo) / 2))
        padding = ((leftPadding, rightPadding), (0, 0), (0, 0))
        inpvideo = np.pad(inpvideo, padding, "constant")
    elif inpLenvideo > inpLenaudio:
        Target_frame = framebeforeConv(inpLenvideo)
        leftPadding = int(np.floor((Target_frame - len(inpaudio))/2))
        rightPadding = int(np.ceil((Target_frame - len(inpaudio))/2))
        padding = ((leftPadding, rightPadding), (0, 0))
        inpaudio = np.pad(inpaudio, padding, "constant")

    inpLenvideo = len(inpvideo)

    reqInpLen = req_input_length(trgt)
    if inpLenvideo < reqInpLen: #
        Target_frame = framebeforeConv(reqInpLen)
        leftPadding = int(np.floor((Target_frame - len(inpaudio))/2))
        rightPadding = int(np.ceil((Target_frame - len(inpaudio))/2))
        inpaudio = np.pad(inpaudio, ((leftPadding, rightPadding), (0, 0)), "constant")

    inpLenvideo = len(inpvideo)
    reqInpLen = max(reqInpLen, inpLenvideo)


    inpvideo = (torch.from_numpy(inpvideo) / 255. - videoParams["NORMALIZATION_MEAN"]) / videoParams["NORMALIZATION_STD"]
    if istrain:
        randomCrop = random.randint(0, 10)
        inpvideo = inpvideo[:, randomCrop: randomCrop+112, randomCrop: randomCrop+112]
        if random.randint(0, 1):
            inpvideo = torch.flip(inpvideo, [-1])
    else:
        inpvideo = inpvideo[:, 5: 117, 5: 117]


    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
    else:
        trgt = None

    inpaudio = torch.from_numpy(inpaudio)
    inpLenvideo = torch.tensor(inpLenvideo)
    inpLenaudio = torch.tensor(inpaudio.shape[0])
    reqInpLen = torch.tensor(reqInpLen)

    return (inpaudio, inpvideo), trgt, (inpLenaudio, inpLenvideo), reqInpLen


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    inputBatch: B*T*1
    """
    inputLenaudio = torch.stack([data[2][0] for data in dataBatch])
    inputLenvideo = torch.stack([data[2][1] for data in dataBatch])
    reqLenBatch = torch.stack([data[3] for data in dataBatch])

    inputaudio = pad_sequence([data[0][0] for data in dataBatch], batch_first=True)
    # for data in dataBatch:
    #     logging.info('{}, {}'.format(data[0][1].shape, type(data[0][1])))
    inputvideo = pad_sequence([data[0][1] for data in dataBatch], batch_first=True)
    if not any(data[1] is None for data in dataBatch):
        # targetBatch = torch.cat([data[1] for data in dataBatch])
        targetBatch = pad_sequence([data[1].unsqueeze(dim=1) for data in dataBatch], batch_first=True, padding_value=-1)
    else:
        targetBatch = None


    return (inputaudio, inputvideo), targetBatch, (inputLenaudio, inputLenvideo), reqLenBatch#, dataBatch[0][4]



def req_input_length(trgt):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen

