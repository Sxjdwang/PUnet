"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil, math
import torch.distributed as dist
import torch.multiprocessing as mp
from espnet_config import espnet_args
from torch.optim.lr_scheduler import LambdaLR

from config import args as argsraw
from models.av_early_single import E2E
from data.lrs2_dataset import LRS2Main, LRS2MixGroup
from utils.metrics import asrMetrics
from data.utilsstft import collate_fn

from train_utils import init_logging, inverseSquareRoot, retrieve, load_parameter, num_params, stepunit, load_video, load_vsr_model


def main(gpu, argfull):
    args = argfull[0]
    esp_args = argfull[1]

    if args["MultiProcess"]:
        rank = gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args["num_gpu"], rank=rank)

    args["device"] = "cuda:" + str(gpu)

    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    # gpuAvailable = False
    device = torch.device(args["device"] if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": False} if gpuAvailable else {}
    if args["MultiProcess"]:
        kwargs["pin_memory"] = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #declaring the train and validation datasets and their corresponding dataloaders
    noisefile = args["CODE_DIRECTORY"] + "/noise.wav"

    noiseParams = {"FILE": noisefile, 'LEVEL': args["SNR"], 'Prob': 0.25}


    audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
    videoParams = {"videoFPS": args["VIDEO_FPS"], "NORMALIZATION_MEAN": args["NORMALIZATION_MEAN"], "NORMALIZATION_STD": args["NORMALIZATION_STD"]}

    trainData = LRS2MixGroup(args["trainset"], args["DATA_DIRECTORY"], args["AUDIO_DIRECTORY"], args["IMAGE_DIRECTORY"], args["BATCH_SIZE"], args["MAIN_REQ_INPUT_LENGTH"],
                                args["CHAR_TO_INDEX"], args["STEP_SIZE"]/args["BATCH_SIZE"], audioParams, videoParams, noiseParams, args["max_len"], "config/specaug.yaml",
                                short_startup=True)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainData, num_replicas=args["num_gpu"], rank=gpu) if args["MultiProcess"] else None
    shuffleset = False if args["MultiProcess"] else True

    trainLoader = DataLoader(trainData, batch_size=1, collate_fn=retrieve, shuffle=shuffleset, sampler=trainSampler, **kwargs)

    noiseParams = {"FILE": noisefile, 'LEVEL': [9999], 'Prob': [1]}

    valData = LRS2Main("val", args["DATA_DIRECTORY"], args["AUDIO_DIRECTORY"], args["IMAGE_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                              audioParams, videoParams, noiseParams, args["Dataset"], "config/specaug.yaml", args["E2E_Training"])
    valSampler = torch.utils.data.distributed.DistributedSampler(valData, num_replicas=args["num_gpu"], rank=gpu) if args["MultiProcess"] else None
    valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, sampler=valSampler, **kwargs)
    #
    #declaring the model, optimizer, scheduler and the loss function
    metrcis = asrMetrics(args["CHAR_TO_INDEX"][" "])

    model = E2E((321, 512), 40, 'trainAudioArg', metrcis, args["E2E_Training"])
    model.to(device)

    load_vsr_model(model, args["PRETRAINED_VIDEO_FILE"], device)

    if args["MultiProcess"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)


    optimizer = optim.Adam(model.parameters(), lr=0.5, betas=(args["MOMENTUM1"], args["MOMENTUM2"]), eps=1e-9,)#args["INIT_LR"]#0.5
    if args["trainset"] == "mixtrain":
        scheduler = LambdaLR(optimizer, inverseSquareRoot)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                         patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"],
                                                         threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)

    if not args["MultiProcess"] or gpu == 0:
        os.makedirs('log', exist_ok=True)
        logger = init_logging(log_name=f'log/{args["name"]}_train.log')

    trainingLossAttCurve = list()
    validationLossAttCurve = list()
    validationLossCTCCurve = list()
    trainingLossCTCCurve = list()
    validationWERCurve = [100]

    #printing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" %(numTotalParams))
    print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

    os.makedirs(args["SAVE_DIRECTORY"] + "/{}/checkpoints".format(args["name"]), exist_ok=True)


    print("\nTraining the model .... \n")

    iteration = 0
    for step in range(args["NUM_STEPS"]):
        iteration = stepunit(args, step, iteration, trainSampler, model, trainLoader, valLoader,
                             optimizer, scheduler, logger, device, gpu,
                             trainingLossAttCurve, trainingLossCTCCurve, validationLossAttCurve,
                             validationLossCTCCurve, validationWERCurve)

    print("\nTraining Done.\n")


if __name__ == "__main__":

    argsraw["trainset"] = 'mixtrain'#"train"#
    esp_args = espnet_args()

    argsraw["IMAGE_DIRECTORY"] = argsraw["DATA_DIRECTORY"]
    argsraw["AUDIO_DIRECTORY"] = argsraw["DATA_DIRECTORY"]

    argsraw["NUM_WORKERS"] = 4
    argsraw["device"] = '0,1' if argsraw["MultiProcess"] else '0'

    argsraw["num_gpu"] = len(argsraw["device"].split(','))

    if argsraw["MultiProcess"]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29511'
        os.environ['CUDA_VISIBLE_DEVICES'] = argsraw["device"]

        mp.spawn(main, nprocs=argsraw["num_gpu"], args=(argsraw, esp_args))
    else:
        main(argsraw["device"], (argsraw, esp_args))
