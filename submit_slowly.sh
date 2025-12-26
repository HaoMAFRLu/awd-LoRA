#!/bin/bash

PARAMS=batch_params.txt
SUBFILE=job.sub
INDEX=0

while IFS=" " read -r arg1 arg2 arg3 arg4 arg5; do
    echo "[INFO] Submit job index $INDEX with rho=$arg1 alpha=$arg2 beta=$arg3 dalpha=$arg4 dbeta=$arg5"

    condor_submit_bid 50 \
         -a "arguments=-u -m torch.distributed.run \
            --nproc_per_node=8 --nnodes=1 \
            --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$((29500 + INDEX)) \
            scripts/train_salad.py --rho $arg1 --alpha_rate $arg2 --beta_rate $arg3 --dalpha $arg4 --dbeta $arg5" \
         -a "log=jobs_batch1/task${INDEX}.log" \
         -a "output=jobs_batch1/task${INDEX}.out" \
         -a "error=jobs_batch1/task${INDEX}.err" \
         "$SUBFILE"

    INDEX=$((INDEX + 1))
    sleep 100
done < "$PARAMS"
