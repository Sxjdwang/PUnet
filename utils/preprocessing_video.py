"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import cv2 as cv
import numpy as np
import torch
import os



def preprocess_sample(file, params):

    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"
    roiFile = file + ".png"
    visualFeaturesFile = file.replace('/LRW/', '/embed/') + ".npy"

    if os.path.exists(visualFeaturesFile):
        return

    if not os.path.exists(os.path.dirname(visualFeaturesFile)):
        os.makedirs(os.path.dirname(visualFeaturesFile))

    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = list()
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed/255
            grayed = cv.resize(grayed, (224,224))
            roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
            roiSequence.append(roi)
        else:
            break
    captureObj.release()
    # cv.imwrite(roiFile, np.floor(255*np.concatenate(roiSequence, axis=1)).astype(np.int))
    """"""
    # import numpy as np
    # import cv2 as cv
    # videoFile = '../mvlrs_v1/main/6330311066473698535/00011.mp4'
    # captureObj = cv.VideoCapture(videoFile)
    # roiSequence = list()
    # roiSize = 122
    # while (captureObj.isOpened()):
    #     ret, frame = captureObj.read()
    #     if ret == True:
    #         grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         grayed = cv.resize(grayed, (224, 224))
    #         roi = grayed[int(112 - (roiSize / 2)):int(112 + (roiSize / 2)),
    #               int(112 - (roiSize / 2)):int(112 + (roiSize / 2))]
    #         roiSequence.append(roi)
    #     else:
    #         break
    # captureObj.release()
    # inp = np.stack(roiSequence, axis=0)
    # np.savez('../mvlrs_v1/main/6330311066473698535/00011.npz', data=inp)
    """"""


    #normalise the frames and extract features for each frame using the visual frontend
    #save the visual features to a .npy file
    if len(roiSequence) == 0:
        print(videoFile)
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1,2])
    inp = (inp - normMean)/normStd
    inputBatch = torch.from_numpy(inp)
    inputBatch = (inputBatch.float()).to(device)
    inputBatch_horizontal = torch.flip(inputBatch, [-1])
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch_horizontal)
    out = torch.squeeze(outputBatch, dim=1)
    out = out.cpu().numpy()
    np.save(visualFeaturesFile, out)
    return


# import cv2 as cv
# import numpy as np
# import torch
# import os
#
# def saveimages(filename):
#     captureObj = cv.VideoCapture(filename)
#     roiSequence = list()
#     roiSize = 224
#     roiFile = 'images/' + filename.replace('.mp4', '')
#     id = 0
#     while (captureObj.isOpened()):
#         ret, frame = captureObj.read()
#         if ret == True:
#             grayed = frame
#             grayed = cv.resize(grayed, (224,224))
#             roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
#             cv.imwrite(roiFile + str(id) + '.png', roi.astype(np.int))
#             id += 1
#             if id >= 10:
#                 break
#         else:
#             break
#     captureObj.release()
#
# saveimages('6326461057659161695_00016.mp4')
# import os.path as path
# import os
# with open('mixtrain_length.txt') as f:
#     lines = f.readlines()
#
# i = 0
# for line in lines:
#     line_list = line.strip().split(' ')
#     line = line_list[0]
#     num = int(line_list[1])
#     set = line.split('/')[0]
#     if set != 'main':
#         continue
#     path_dir = 'wav2lip/'+line.replace('main/', '')
#     if not path.exists(path_dir):
#         print('not'+line)
#     else:
#         files = os.listdir(path_dir)  # dir is your directory path
#         if abs(num - len(files)) > 0:
#             i += 1
#             print(line, ' ', abs(num - len(files)), ' ', num, len(files))
# print(i)
