"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import os
import math
import torch
from tqdm import tqdm

import numpy as np
import torch.distributed as dist


def reduce_sum(data, device):
    tensor = torch.tensor(data).to(device, non_blocking=True)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def train(model, trainLoader, optimizer, device, mp=1, step=None, schedule=None, logger=None, update=4):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLossatt = 0
    trainingLossctc = 0

    forward_count = 0
    acc = 0
    updata = 200
    model.train()
    optimizer.zero_grad()

    for batch, ((inputAudio, inputVideo), targetBatch, (inputLenaudio, inputLenvideo), reqLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):

        inputAudio, inputVideo = (inputAudio).to(device, non_blocking=True, dtype=torch.float32), (inputVideo).to(device, non_blocking=True, dtype=torch.float32)

        inputLenaudio, inputLenvideo = (inputLenaudio.int()).to(device, non_blocking=True), (inputLenvideo.int()).to(device, non_blocking=True)
        reqLenBatch, targetBatch = (reqLenBatch.int()).to(device, non_blocking=True), (targetBatch.long()).to(device, non_blocking=True)

        loss, _, (loss_att, loss_ctc) = model(inputAudio, inputVideo, inputLenaudio, inputLenvideo, reqLenBatch, targetBatch.squeeze(dim=2))

        if mp > 1:
            torch.distributed.barrier()
            reduce_loss_att = reduce_sum(loss_att * inputAudio.shape[0], device)
            reduce_loss_ctc = reduce_sum(loss_ctc * inputAudio.shape[0], device)
        else:
            reduce_loss_att = loss_att * inputAudio.shape[0]
            reduce_loss_ctc = loss_ctc * inputAudio.shape[0]

        loss.backward()
        trainingLossatt = trainingLossatt + reduce_loss_att
        trainingLossctc = trainingLossctc + reduce_loss_ctc
        forward_count += 1

        if forward_count < update:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.
        )
        if math.isnan(grad_norm):
            print("grad norm is nan. Do not update model.")
        else:
            optimizer.step()
        optimizer.zero_grad()
        forward_count = 0

        if schedule is not None:
            schedule.step(step)
            step += 1
            acc += model.acc
            if int(step) % updata == 0:
                logger.info('{:04d} iterations, accuracy is {:f}'.format(step, acc/updata))
                acc = 0

    if forward_count != 0:
        optimizer.step()
        optimizer.zero_grad()

    trainingLossatt = trainingLossatt / len(trainLoader.dataset)
    trainingLossctc = trainingLossctc / len(trainLoader.dataset)

    return trainingLossatt, trainingLossctc, step


def train_simu(model, trainLoader, optimizer, device, mp=1, step=None, schedule=None, logger=None):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLossatt = 0
    trainingLossctc = 0

    forward_count = 0
    acc = 0
    updata = 200
    model.train()

    for batch, (inputAudio, targetBatch, inputLenBatch, inputVideo) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):

        inputAudio, inputVideo = (inputAudio).to(device, non_blocking=True, dtype=torch.float32), (inputVideo).to(device, non_blocking=True, dtype=torch.float32)

        inputLenBatch, targetBatch = (inputLenBatch.int()).to(device, non_blocking=True), (targetBatch.long()).to(device, non_blocking=True)

        loss, _, (loss_att, loss_ctc) = model(inputAudio, inputVideo, inputLenBatch, targetBatch.squeeze(dim=2))

        if mp > 1:
            torch.distributed.barrier()
            reduce_loss_att = reduce_sum(loss_att * inputAudio.shape[0], device)
            reduce_loss_ctc = reduce_sum(loss_ctc * inputAudio.shape[0], device)
        else:
            reduce_loss_att = loss_att * inputAudio.shape[0]
            reduce_loss_ctc = loss_ctc * inputAudio.shape[0]

        loss.backward()
        trainingLossatt = trainingLossatt + reduce_loss_att
        trainingLossctc = trainingLossctc + reduce_loss_ctc

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.
        )
        if math.isnan(grad_norm):
            print("grad norm is nan. Do not update model.")
        else:
            if opmode == 'AO':
                optimizer['audio'].step()
            elif opmode == 'VO':
                optimizer['video'].step()
            else:
                optimizer['video'].step()
                optimizer['audio'].step()
            optimizer['rest'].step()
        for key in optimizer.keys():
            optimizer[key].zero_grad()

        if schedule is not None:
            for key in schedule.keys():
                schedule[key].step(step)
            step += 1
            acc += model.acc
            if int(step) % updata == 0:
                logger.info('{:04d} iterations, accuracy is {:f}'.format(step, acc/updata))
                acc = 0

    trainingLossatt = trainingLossatt / len(trainLoader.dataset)
    trainingLossctc = trainingLossctc / len(trainLoader.dataset)

    return trainingLossatt, trainingLossctc, step



def evaluate(model, evalLoader, device, mp=1):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss_att = 0
    evalLoss_ctc = 0
    evalCER = 0
    evalWER = 0
    model.eval()

    for batch, ((inputAudio, inputVideo), targetBatch, (inputLenaudio, inputLenvideo), reqLenBatch) in enumerate(
            tqdm(evalLoader, leave=False, desc="eval",
                 ncols=75)):
        inputAudio, inputVideo = (inputAudio).to(device, non_blocking=True, dtype=torch.float32), (inputVideo).to(
            device, non_blocking=True, dtype=torch.float32)

        inputLenaudio, inputLenvideo = (inputLenaudio.int()).to(device, non_blocking=True), (
            inputLenvideo.int()).to(device, non_blocking=True)
        reqLenBatch, targetBatch = (reqLenBatch.int()).to(device, non_blocking=True), (targetBatch.long()).to(
            device, non_blocking=True)

        with torch.no_grad():
            loss, (cer, wer), (loss_att, loss_ctc) = model(inputAudio, inputVideo, inputLenaudio, inputLenvideo, reqLenBatch, targetBatch.squeeze(dim=2))

        if mp > 1:
            torch.distributed.barrier()
            reduce_loss_att = reduce_sum(loss_att * inputAudio.shape[0], device)
            reduce_loss_ctc = reduce_sum(loss_ctc * inputAudio.shape[0], device)
            reduce_cer = reduce_sum(cer * inputAudio.shape[0], device)
            reduce_wer = reduce_sum(wer * inputAudio.shape[0], device)
        else:
            reduce_loss_att = loss_att * inputAudio.shape[0]
            reduce_loss_ctc = loss_ctc * inputAudio.shape[0]
            reduce_cer = cer * inputAudio.shape[0]
            reduce_wer = wer * inputAudio.shape[0]

        evalLoss_att = evalLoss_att + reduce_loss_att
        evalLoss_ctc = evalLoss_ctc + reduce_loss_ctc

        evalCER = evalCER + reduce_cer
        evalWER = evalWER + reduce_wer

    evalLoss_att = evalLoss_att / len(evalLoader.dataset)
    evalLoss_ctc = evalLoss_ctc / len(evalLoader.dataset)
    evalCER = evalCER/len(evalLoader.dataset)
    evalWER = evalWER/len(evalLoader.dataset)
    if mp > 1:
        torch.distributed.barrier()
        evalWER = reduce_sum(evalWER, device).item() / mp
    return evalLoss_att, evalLoss_ctc, evalCER, evalWER


def testmodel_scorer(model, beam_search, charlist, evalLoader, device, args, rank=0):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss_att = 0
    evalLoss_ctc = 0
    evalCER = 0
    evalWER = 0

    model.eval()
    beam_search.eval()

    root = 'exp_decoding'
    os.makedirs(f'{root}/{args.name}', exist_ok=True)

    hypo_file = open(f'{root}/{args.name}/hypo_{rank}.txt', 'w')
    gt_file = open(f'{root}/{args.name}/gt_{rank}.txt', 'w')

    for batch, ((inputAudio, inputVideo), targetBatch, (inputLenaudio, inputLenvideo), reqLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                          ncols=75)):

        inputAudio, inputVideo = (inputAudio).to(device, non_blocking=True, dtype=torch.float32), (inputVideo).to(
            device, non_blocking=True, dtype=torch.float32)

        inputLenaudio, inputLenvideo = (inputLenaudio.int()).to(device, non_blocking=True), (
            inputLenvideo.int()).to(device, non_blocking=True)
        reqLenBatch, targetBatch = (reqLenBatch.int()).to(device, non_blocking=True), (targetBatch.long()).to(
            device, non_blocking=True)

        with torch.no_grad():
            enc = model.encode(inputAudio, inputVideo, inputLenaudio, inputLenvideo, reqLenBatch)
            hypo = beam_search(
                x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
            )

        target = ''
        hypos = ''
        for i in range(targetBatch.shape[1]):
            target += charlist[int(targetBatch[0,i,0])]
        gt_file.write(target+'\n')
        for i in range(len(hypo[0].yseq)):
            hypos += charlist[int(hypo[0].yseq[i])]
        hypo_file.write(hypos + '\n')

    hypo_file.close()
    gt_file.close()

    return evalLoss_att, evalLoss_ctc, evalCER, evalWER

