#!/bin/bash

singularity exec --nv --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache:/cache,/data/users/maqsood/thesis/pretrained:/pretrained /nethome/mmshaik/thesis/cross_domain_exp/audio_finetune.sif bash /nethome/mmshaik/thesis/cross_domain_exp/submit.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log
