#!/bin/bash

PARAMS=batch_params.txt
SUBFILE=job.sub
INDEX=0

while IFS=" " read -r arg1 arg2; do
    echo "[INFO] Submit job index $INDEX with alpha=$arg1 beta=$arg2"

    condor_submit_bid 100 \
         -a "arguments=-u -m torch.distributed.run \
            --nproc_per_node=5 --nnodes=1 \
            --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$((29500 + INDEX)) \
            scripts/train_salad.py --alpha_rate $arg1 --beta_rate $arg2" \
         -a "log=jobs_batch1/task${INDEX}.log" \
         -a "output=jobs_batch1/task${INDEX}.out" \
         -a "error=jobs_batch1/task${INDEX}.err" \
         "$SUBFILE"

    INDEX=$((INDEX + 1))
    sleep 100
done < "$PARAMS"
