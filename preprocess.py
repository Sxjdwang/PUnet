"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
import torchaudio

from config import args
from utils.preprocessing_audio import preprocess_sample


def main():

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])

    # with open(f'{args["DATA_DIRECTORY"]}/test.txt', 'r') as f:
    #     lines = f.readlines()
    #
    # filesList = list()
    # for line in lines:
    #     filesList.append(f"{args['DATA_DIRECTORY']}/main/{line.strip().split(' ')[0]}")

    #walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4]))


    # #Preprocessing each sample
    # print("\nNumber of data samples to be processed = %d" %(len(filesList)))
    # print("\n\nStarting preprocessing ....\n")
    #
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file)

    print("\nPreprocessing Done.")

    #Generating a 1 hour noise file
    #Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
    #The length of these clips is the shortest audio sample among the 20 samples
    print("\n\nGenerating the noise file ....")
    # filesList = [] # the list of all wavform files
    noise = np.empty((0))
    while len(noise) < 16000*3600:
        noisePart = np.zeros(16000*60)
        indices = np.random.randint(0, len(filesList), 20)
        for ix in indices:
            sampFreq, audio = wavfile.read(filesList[ix] + ".wav")
            audio = audio/np.max(np.abs(audio))
            pos = np.random.randint(0, abs(len(audio)-len(noisePart))+1)
            if len(audio) > len(noisePart):
                noisePart = noisePart + audio[pos:pos+len(noisePart)]
            else:
                noisePart = noisePart[pos:pos+len(audio)] + audio
        noise = np.concatenate([noise, noisePart], axis=0)
    noise = noise[:16000*3600]
    noise = (noise/20)*32767
    noise = np.floor(noise).astype(np.int16)
    wavfile.write(args["DATA_DIRECTORY"] + "/noise.wav", 16000, noise)

    print("\nNoise file generated.")

    return


def append_wav_duration(input_txt, new_lines):
    with open(input_txt, "r") as f:
        lines = f.readlines()

    prefix = 'pretrain' if 'pretrain' in input_txt else 'main'

    for line in lines:
        # wav_path = f"{prefix}/{line.strip()}"
        wav_path = f"{prefix}/{line.split(' ')[0]}"

        try:
            waveform, sample_rate = torchaudio.load(f'{args["DATA_DIRECTORY"]}/{wav_path}.wav')
            duration = int(waveform.shape[1] / sample_rate *25)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            duration = -1

        new_line = f"{wav_path} {duration}"
        new_lines.append(new_line)



if __name__ == "__main__":
    main()

    """Combine pretrain and train set and count lengths of all files"""
    combined_list = []
    for file in os.listdir(args["DATA_DIRECTORY"]):
        if not file.endswith(".txt"):
            continue
        if file == 'test.txt' or file == 'valid.txt':
            continue
        # if file != 'test.txt':
        #     continue
        append_wav_duration(f'{args["DATA_DIRECTORY"]}/{file}', combined_list)
    with open(f'{args["DATA_DIRECTORY"]}/mixtrain_length.txt', "w") as f:
        for l in combined_list:
            f.write(l + "\n")