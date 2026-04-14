#!/usr/bin/env bash

num_thread=3

export PATH=/data/chi-gpu4/jiadong/PUnet:$PATH

snr=9999
run.pl JOB=1:$num_thread exp_tmp/$snr.JOB.log \
 python test.py --ckpt ckpt/ave3f4_16.pt --rank JOB --nshard $num_thread --snr $snr
