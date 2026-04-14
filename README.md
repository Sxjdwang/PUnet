# P&U net

This repo is the official implementation of 'Predict-and-update network: Audio-visual speech recognition inspired by human speech perception', TASLP 2024.

[Paper](https://ieeexplore.ieee.org/abstract/document/10768989)

## Environment

```
conda create -n punet python=3.7
conda activate punet
pip install numpy==1.21.6 torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

## Dataset download 

1. Please download lrs2 dataset from https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html
2. Unzip it and organize the dataset following structure:

```
data_root/
├── main/
├── pretrain/
├── pretrain.txt
├── test.txt
├── train.txt
└── val.txt
```

## Preprocessing

```
python preprocess.py --data_root $data_root
```

## Preprocessing

```
save_exp_name=xxx
python train.py --batch_size 8 --data_root /data/chi-gpu4/jiadong/lrs2/mvlrs_v1 --name $save_exp_name
```
Usually, the CTC module will output some information about its inputs and outputs.
If you want to remove them, you can comment them in xxx/punet/lib/python3.7/site-packages/espnet/nets/pytorch_backend/ctc.py

## Evaluation
```
python averageCheckpoint.py --source_exp $save_exp_name --out_file ${save_exp_name}_combine.pt
python test.py --ckpt ${save_exp_name}_combine.pt --snr 0 or 9999
python wertext.py
```
averageCheckpoint.py aims to average the checkpoints of multiple steps, which might improve performance. You could ignore this steps if not helpful. 

## Trained checkpoints

----------
|         Model          |         Description         |  Link  | 
|:----------------------:|:---------------------------:| :---------------: |
|        punet.pt        | Trained parameters on LRS2  | [Link](https://drive.google.com/file/d/1IgMJEKGa4GG4FFHhMX9BcEdQ2A7Hp_cj/view?usp=drive_link)  |
|     rnnlm.val5.avg.best      |       Language Model        | [Link](https://drive.google.com/file/d/17pms4p0vkxBKX3_xH4KOhdyCeXTB97lL/view?usp=drive_link) |
| model.json |  Langauge model json file   | [Link](https://drive.google.com/file/d/1N8sfZQZc87dHqXhHri2SMT7rkTrYqBs2/view?usp=drive_link) |
| video_pretraining.pt | Pretrained lipreading model | [Link](https://drive.google.com/file/d/1843WG2ZAvUceK159mySpLW2vNDAg3KSM/view?usp=drive_link) |

The model is trained in 2022. The extracted audio with ffmpeg is slightly different from the current version. If you want to test our trained model, please add the following line to data/utilsstft.py (between 125th-127th lines), and use the noise file in the repo.
```
inputAudio = np.concatenate((np.zeros(1024), inputAudio), axis=0)
```


