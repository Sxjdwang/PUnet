import random
import numpy as np
import torch
from scipy.io import wavfile
import sys

__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance']


def NormalizeUtterance(signal):

    signal_std = 0. if torch.std(signal)==0. else torch.std(signal)
    if signal_std == 0.:
        np.save('invalid.npy', signal.numpy())
        # sys.exit('0 std')
    signal_mean = torch.mean(signal)
    return (signal - signal_mean) / signal_std



def Timemasking(singal, max_second, rate):

    max_frames = int(rate * max_second)

    length = np.random.randint(0, max_frames)

    start = np.random.randint(0, singal.size(1) - length)
    end = start + length
    # print('Time mask range', start, end)

    singal[:, start:end].zero_()
    return singal


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise_file, snr_levels=[-5, 0, 5, 10, 15, 20, 9999], snr_prob=None):
        if "wav" in noise_file:
            sr, noise = wavfile.read(noise_file)
        elif "npy":
            noise = np.load(noise_file)
        assert noise.dtype in [np.float32, np.float64, np.int16, np.int32], "noise only supports float data type"
        
        self.noise = torch.from_numpy(noise).unsqueeze(dim=0)
        self.snr_levels = np.array(snr_levels)
        if snr_prob is not None:
            self.prob = np.array(snr_prob)
            self.prob = self.prob / np.sum(self.prob)

    def get_power(self, clip):
        """

        :param clip: (1, N) N is frame number
        :return:
        """
        clip2 = clip.clone().to(torch.int64)
        clip2 = clip2 ** 2
        return torch.sum(clip2) / (clip2.shape[1] * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [torch.int16, torch.int32], "signal only supports int data type"
        # snr_target = random.choice(self.snr_levels)
        snr_target = np.random.choice(self.snr_levels, p=self.prob)
        # print('snr', snr_target)
        if snr_target == 9999:
            return signal#, 0, 0
        else:
            # -- get noise
            start_idx = random.randint(0, self.noise.shape[1]-signal.shape[1])
            noise_clip = self.noise[:, start_idx:start_idx+signal.shape[1]]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor))
            if torch.max(desired_signal) > 65536/2 or torch.min(desired_signal) < -65536/2:
                print(torch.max(desired_signal), torch.min(desired_signal))
            # print(snr_target)
            return desired_signal.to(torch.int16)#, start_idx, noise_clip


def alignedFPS(inp):

    inpLen = len(inp)

    residule = inpLen % 640

    rightpadding = 640 - residule if residule > 0 else 0

    if rightpadding > 0:
        inp = np.pad(inp.numpy(), ((0, rightpadding)), "constant")
        inp = torch.from_numpy(inp)

    final_size = (inpLen + rightpadding) / 640

    return inp, final_size
