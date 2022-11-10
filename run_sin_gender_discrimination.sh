#!/bin/bash

singularity exec --nv --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache:/cache,/data/users/maqsood/thesis/pretrained:/pretrained,/data/users/maqsood/thesis/music_pretrained:/music_pretrained,/data/users/maqsood/thesis/wop:/wop,/data/users/maqsood/thesis/plots:/plots /nethome/mmshaik/thesis/cross_domain_exp/audio_finetune.sif bash /nethome/mmshaik/thesis/cross_domain_exp/submit_gender_accuracies.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log
