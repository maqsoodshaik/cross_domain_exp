#!/bin/bash

singularity exec --nv  --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache_wop:/cache,/data/users/maqsood/thesis/wop:/wop /nethome/mmshaik/thesis/cross_domain_exp/audio_finetune.sif bash /nethome/mmshaik/thesis/cross_domain_exp/submit_wop.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log
