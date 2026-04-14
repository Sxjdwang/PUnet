import os
import torch
from collections import OrderedDict
import argparse
args = dict()

parser = argparse.ArgumentParser(
        description='Code to train the Wav2Lip model WITH the visual quality discriminator')

# dataset
parser.add_argument('--source_exp', help='experiment name for saving', default='result', type=str)
parser.add_argument('--out_file', help='experiment name for saving', default='result', type=str)

args_feed_in = parser.parse_args()

checkDir = f'exp_results/{args_feed_in.source_exp}/checkpoints/'
suffix1 = f'{args_feed_in.source_exp}train-step_0'

epochList = list(range(300, 501, 50))
checkpointList = []

print(epochList)

for i in epochList:
    filename1 = suffix1 + str(i)
    for file in os.listdir(checkDir):
        if filename1 in file:
            checkpointList.append(checkDir + file)

checkpointList.append(checkDir + f'{args_feed_in.source_exp}train-best.pt')
print(checkpointList)

weights = []
for i in checkpointList:
    weights.append(torch.load(i, map_location='cpu'))

Target = OrderedDict()
for key, value in weights[0].items():
    Target[key] = 0
    for i in range(len(weights)):
        Target[key] += weights[i][key]
    Target[key] = Target[key] / float(len(weights))

torch.save(Target, args_feed_in.out_file)
print('Combine checkpoint is saved to', args_feed_in.out_file)



