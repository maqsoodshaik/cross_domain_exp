#!/usr/bin/env bash

# specify which GPU to work on ...
export CUDA_VISIBLE_DEVICES=1

nvidia-smi
# obtain the directory the bash script is stored in


# DIR=$(cd $(dirname $0); pwd)
#--bind /data/corpora:/corpora
#--bind /data/users/maqsood/hf_cache:/cache
export HF_DATASETS_DOWNLOADED_DATASETS_PATH='/corpora/multilingual_librispeech/'
export HF_DATASETS_CACHE='/cache'
python -u ~/thesis/cross_domain_exp/gender_discrimination.py