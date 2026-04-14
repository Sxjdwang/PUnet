"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.visual_frontend import ResNet


class VisualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, minlen=None):
        super(VisualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm2d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
                        )
        self.resnet = ResNet()
        self.minlen = minlen
        return

    def outpadding(self, output, inputLen, lenReq):

        """
        output:a list of (T*512) T is changable
        inputLen: 1d tensor
        lenReq: 1d tensor
        """

        leftpadding = torch.floor((lenReq - inputLen).float()/2).int()
        rightpadding = torch.max(lenReq) - leftpadding - inputLen #+ 3
        # lenReq += 3
        for i in range(len(output)):
            output[i] = F.pad(output[i], (0, 0, leftpadding[i], rightpadding[i])).unsqueeze(dim=0)
        return torch.cat(output, dim=0)


    def forward(self, inputBatch, inputLen, lenReq):
        # data = []
        batchsize = inputBatch.shape[0]

        batch = self.frontend3D[0](inputBatch).transpose(1, 2)
        # data.append(batch.detach().clone())
        batch2d = torch.cat([batch[i, :inputLen[i]] for i in range(inputLen.shape[0])], dim=0)
        batch = self.frontend3D[1:](batch2d)

        # data.append(batch.detach().clone())

        outputResnet = self.resnet(batch).squeeze(dim=3).squeeze(dim=2)

        paddingList = outputResnet.split(inputLen.cpu().detach().numpy().tolist(), dim=0)
        outputBatch = self.outpadding(list(paddingList), inputLen, lenReq)

        assert batchsize == outputBatch.shape[0]
        return outputBatch, lenReq#, data
