#!/bin/bash

singularity exec --nv --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache:/cache,/data/users/maqsood/thesis/plots:/plots,/data/users/maqsood/thesis/pretrained:/pretrained,/data/users/maqsood/thesis/wop:/wop /nethome/mmshaik/thesis/inlp_exp/inlp_exp.sif bash /nethome/mmshaik/thesis/cross_domain_exp/submit_eval_monolingual.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log
