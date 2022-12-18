#!/usr/bin/env bash

# specify which GPU to work on ...
export CUDA_VISIBLE_DEVICES=3

nvidia-smi
# obtain the directory the bash script is stored in


# DIR=$(cd $(dirname $0); pwd)
#--bind /data/corpora:/corpora
#--bind /data/users/maqsood/hf_cache:/cache
export HF_DATASETS_DOWNLOADED_DATASETS_PATH='/corpora/common_voice_speech/'
export HF_DATASETS_CACHE='/cache'
python -u ~/thesis/cross_domain_exp/LID_eval.py 