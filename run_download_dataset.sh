singularity exec --nv --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache:/cache /nethome/mmshaik/thesis/audio_finetune.sif bash /nethome/mmshaik/thesis/submit_download.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log