"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models.av_early_single import E2E
import os, shutil
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.beam_search import BeamSearch

from config import args as argsraw
from data.lrs2_dataset import LRS2Main
from utils.general import testmodel_scorer, evaluate
from utils.metrics import asrMetrics

from train_utils import init_logging

from data.utilsstft import collate_fn

def main(args, noisefile):

    # matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    # gpuAvailable = False
    device = torch.device(args["device"] if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #declaring the pretrain and the preval datasets and the corresponding dataloaders
    audioParams = {"stftWindow": args["STFT_WINDOW"], "stftWinLen": args["STFT_WIN_LENGTH"],
                   "stftOverlap": args["STFT_OVERLAP"]}
    videoParams = {"videoFPS": args["VIDEO_FPS"], "NORMALIZATION_MEAN": args["NORMALIZATION_MEAN"],
                   "NORMALIZATION_STD": args["NORMALIZATION_STD"]}

    noiseParams = {"FILE": noisefile, 'LEVEL': [args["NOISE_SNR_DB"]], 'Prob': [1]}

    valData = LRS2Main("test", args["DATA_DIRECTORY"], args["AUDIO_DIRECTORY"], args["IMAGE_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                              audioParams, videoParams, noiseParams, args["Dataset"], "config/specaug.yaml", args["E2E_Training"], rank=args["rank"], nshard=args["nshard"], ap=args["ap"])
    valLoader = DataLoader(valData, batch_size=1, collate_fn=collate_fn, shuffle=False, **kwargs)

    #declaring the model, optimizer, scheduler and the loss function
    metrcis = asrMetrics(args["CHAR_TO_INDEX"][" "])

    model = E2E((321, 512), 40, 'trainAudioArg', metrcis, args["E2E_Training"], vcue_dim=args["v2a_dim"], addffn=args["add_ffn"])

    model.eval()
    model.to(device)

    model.load_state_dict(torch.load(args["TRAINED_MODEL_FILE"], map_location=device))

    if args["TRAINED_LM_FILE"] and args['target'] == 'decoder':
        print("loading language model", args["TRAINED_LM_FILE"])
        from espnet.asr.asr_utils import get_model_conf
        from espnet.nets.lm_interface import dynamic_import_lm
        from espnet.asr.asr_utils import torch_load

        lm_args = get_model_conf(args["TRAINED_LM_FILE"], args["TRAINED_LM_FILE"].replace('rnnlm.val5.avg.best', 'model.json').replace('lmlrs3avg.best', 'model.json'))
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(40, lm_args)
        torch_load(args["TRAINED_LM_FILE"], lm)
        lm.eval()
    else:
        lm = None

    recog_args_path = 'config/decode_config.yaml'

    recog_args = model.argsetup(recog_args_path, ": ")
    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(40)
    weights = dict(
        decoder=1.0 - recog_args.ctc_weight,
        ctc=recog_args.ctc_weight,
        lm=recog_args.lm_weight,
        length_bonus=recog_args.penalty,
    )
    beam_search = BeamSearch(
        beam_size=recog_args.beam_size,
        vocab_size=model.odim,
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=args["Char_List"],
        pre_beam_score_key=None if recog_args.ctc_weight == 1.0 else "full",
    )
    beam_search.eval()
    beam_search.to(device)

    #printing the total and trainable parameters in the model

    print("Number of Words = %d" %(args["PRETRAIN_NUM_WORDS"]))
    print("\nPretraining the model .... \n")



    recog_args.name = '{}_noise_{}dB'.format(args["Dataset"], args["NOISE_SNR_DB"])
    validationLoss_att, validationLoss_ctc, validationCER, validationWER = testmodel_scorer(model, beam_search, args["INDEX_TO_CHAR"], valLoader, device, recog_args, args["rank"])

    return validationWER


if __name__ == "__main__":

    argsraw["IMAGE_DIRECTORY"] = argsraw["DATA_DIRECTORY"]
    argsraw["AUDIO_DIRECTORY"] = argsraw["DATA_DIRECTORY"]

    argsraw['name'] = 'decode'

    argsraw['mode'] = 'scorer'

    argsraw['target'] = 'decoder'

    argsraw["NUM_WORKERS"] = 0

    argsraw["device"] = "cuda:" + argsraw["device"]

    main(argsraw, 'noise.wav')


