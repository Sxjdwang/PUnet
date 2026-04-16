import torch
import logging
import subprocess
import numpy as np
from utils.general import train, evaluate, train_simu


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def init_logging(level=logging.INFO,
                 log_name='log/sys.log',
                 formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')):
    logger = logging.getLogger()
    logger.setLevel(level=level)
    handler = logging.FileHandler(log_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def inverseSquareRoot(step):
    # one epoch comsume 16384 samples
    # benchmark paper take 2e5 samples to warm up from 1e-4 to 4e-4
    # for epoch
    step = max(step, 1)
    model_size = 256
    warmup = 25000
    return (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def retrieve(data):
    return data[0][0], data[0][1], data[0][2], data[0][3]


def load_parameter(model, pretrain_audio, pretrain_video, device):
    print('load audio and video paramters from', pretrain_audio, pretrain_video)
    audio_param = torch.load(pretrain_audio, map_location=device)
    video_param = torch.load(pretrain_video, map_location=device)

    model_dict = model.state_dict()

    for key, value in audio_param.items():
        if 'encoder' in key:
            if 'audio'+key in model_dict.keys():
                model_dict['audio'+key] = audio_param[key]
            else:
                print(key)
        # else:
        #     model_dict[key] = audio_param[key]

    for key, value in video_param.items():
        if 'encoder' in key:
            model_dict['video' + key] = video_param[key]
        else:
            model_dict[key] = video_param[key]
    model.load_state_dict(model_dict)
    model.to(device)


def load_vsr_model(model, pretrain_video, device):
    print('load video paramters from', pretrain_video)
    video_param = torch.load(pretrain_video, map_location=device)

    model_dict = model.state_dict()

    for key, value in video_param.items():
        if 'encoder' in key:
            model_dict['video' + key] = video_param[key]
            if 'audio' + key in model_dict.keys():
                if model_dict['audio' + key].shape == video_param[key].shape:
                    model_dict['audio' + key] = video_param[key]
            #     else:
            #         print('the configuration of pretrained model is different from the current in', key)
            # else:
            #     print(key, 'is not in current model')
        else:
            model_dict[key] = video_param[key]
    model.load_state_dict(model_dict)
    model.to(device)



def load_video(model, pretrain_video, device):
    print('load video paramters from', pretrain_video)
    video_param = torch.load(pretrain_video, map_location=device)

    model_dict = model.state_dict()

    for key, value in video_param.items():
        if 'encoder' in key:
            model_dict['video' + key] = video_param[key]
        elif 'visual_frontend' in key:
            model_dict[key] = video_param[key]
        elif 'ctc.ctc_lo' in key:
            model_dict[key.replace('ctc.ctc_lo', 'projv2a')] = video_param[key]

    model.load_state_dict(model_dict)
    model.to(device)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[0]


def stepunit(args, step, iteration, trainSampler, model, trainLoader, valLoader, optimizer, scheduler, logger, device, gpu,
             trainingLossAttCurve, trainingLossCTCCurve, validationLossAttCurve, validationLossCTCCurve, validationWERCurve):
    # train the model for one step
    if args["MultiProcess"]:
        trainSampler.set_epoch(step)

    update = 4 if args["E2E_Training"]['video'] else 2

    if args["trainset"] == "mixtrain":
        trainingLoss_att, trainingLoss_ctc, iteration = train(model, trainLoader, optimizer, device, args["num_gpu"],
                                                              step=iteration, schedule=scheduler, logger=logger, update=update)
    else:
        trainingLoss_att, trainingLoss_ctc, _ = train(model, trainLoader, optimizer, device, args["num_gpu"])
        # trainingLoss_att, trainingLoss_ctc = 0, 0
    trainingLossAttCurve.append(trainingLoss_att)
    trainingLossCTCCurve.append(trainingLoss_ctc)

    # evaluate the model on validation set
    validationLoss_att, validationLoss_ctc, validationCER, validationWER = evaluate(model, valLoader, device,
                                                                                    args["num_gpu"])
    # validationLoss_att, validationLoss_ctc, validationCER, validationWER = 0, 0, 0, 0
    validationLossAttCurve.append(validationLoss_att)
    validationLossCTCCurve.append(validationLoss_ctc)

    if not args["MultiProcess"] or gpu == 0:
        # printing the stats after each step
        print(
            "Step: %03d || Tr.Loss_att: %.6f  Tr.Loss_ctc: %.6f || val.Loss_att: %.3f  Val.Loss_ctc: %.3f || Val.CER: %.3f  Val.WER: %.3f || gpu: %d"
            % (step, trainingLoss_att, trainingLoss_ctc, validationLoss_att, validationLoss_ctc, validationCER,
               validationWER, get_gpu_memory_map()))
        # import sys
        # sys.exit()
        logger.info('{:04d} epoch {:04d} iterations learning rate is {:f}'.format(step, iteration,
                                                                                  optimizer.param_groups[0]['lr']))
        logger.info(
            "Step: {:03d} || Tr.Loss_att: {:.3f}  Tr.Loss_ctc: {:.3f}".format(step, trainingLoss_att, trainingLoss_ctc))
        logger.info("Step: {:03d} || val.Loss_att: {:.3f}  Val.Loss_ctc: {:.3f}".format(step, validationLoss_att,
                                                                                        validationLoss_ctc))
        logger.info("Step: {:03d} || Val.CER: {:.3f}  Val.wer: {:.3f} || gpu: {}".format(step, validationCER, validationWER, get_gpu_memory_map()))
        logger.info(' ')

    # make a scheduler step
    if args["trainset"] != "mixtrain":
        scheduler.step(validationWER)

    if step >= 100:
        trainLoader.dataset.shortfirst = False
    trainLoader.dataset.choose_section()
    if np.sum(trainLoader.dataset.roundprob) == 0:
        trainLoader.dataset.reformbatch()

    if not args["MultiProcess"] or gpu == 0:
        if validationWER < min(validationWERCurve):
            savePath = args["SAVE_DIRECTORY"] + "/{}/checkpoints/{:s}train-best.pt".format(args["name"], args["name"])
            torch.save(model.state_dict(), savePath)

        savePath = args["SAVE_DIRECTORY"] + "/{}/checkpoints/{:s}train-last.pt".format(args["name"], args["name"])
        torch.save(model.state_dict(), savePath)

    validationWERCurve.append(validationWER)

    #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
    if ((step % args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"] - 1))\
            and (step != 0) and args["trainset"] == "mixtrain":

        if not args["MultiProcess"] or gpu == 0:
            savePath = args["SAVE_DIRECTORY"] + "/{}/checkpoints/{:s}train-step_{:04d}-wer_{:.3f}.pt".format(args["name"],
                args["name"], step, validationWER)
            torch.save(model.state_dict(), savePath)

    return iteration
