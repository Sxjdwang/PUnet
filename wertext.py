import numpy as np
import editdistance
import argparse
import os

def process_line(text):
    new_line = ''
    for i in text:
        if i == '(':
            break
        new_line += i
    return new_line

def calculateWer(gtfile, hypofile, espnet=False):

    spaceIx = ' '

    with open(gtfile, 'r') as f:
        trgts = f.readlines()

    with open(hypofile, 'r') as f:
        preds = f.readlines()

    totalEdits = 0
    totalWords = 0

    Edit_list = []

    wers = []


    for n in range(len(preds)):
        if espnet:
            pred = process_line(preds[n])
            trgt = process_line(trgts[n])
        else:
            pred = preds[n].strip()
            trgt = trgts[n].strip()

            pred = pred.replace('<EOS>', '')

        predWords = pred.split(spaceIx)

        trgtWords = trgt.split(spaceIx)

        if espnet:
            predWords = predWords[:-1]
            trgtWords = trgtWords[:-1]

        numEdits = editdistance.eval(predWords, trgtWords)
        totalEdits = totalEdits + numEdits
        totalWords = totalWords + len(trgtWords)
        wers.append(float(numEdits)/len(trgtWords))


        Edit_list.append(numEdits)

    print("total distance %d in total words %d, WER %f"%(totalEdits, totalWords, totalEdits/totalWords))

    return Edit_list, totalEdits, totalWords

parser = argparse.ArgumentParser(
        description='Code to train the Wav2Lip model WITH the visual quality discriminator')
# dataset
parser.add_argument('--name', help='batch size of training', default=None, type=str)

args = parser.parse_args()

root = 'exp_decoding'
for file in os.listdir(root):
    print('check ', file)
    edit, words = 0, 0
    for j in range(4):
        if not os.path.exists('{}/{}/gt_{}.txt'.format(root, file, j)):
            continue
        ownedit1, edit1, words1 = calculateWer('{}/{}/gt_{}.txt'.format(root, file, j), '{}/{}/hypo_{}.txt'.format(root, file, j))
        edit += edit1
        words += words1
    print("total distance %d in total words %d, WER %f"%(edit, words, (edit)/(words)))
