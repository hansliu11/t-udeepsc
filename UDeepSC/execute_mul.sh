#!/bin/bash

# Define args
ST=17
ED=24
GPU=0

for (( i=${ST}; i<=${ED}; i=i+1 )); do
    ns=${i}
    echo "Running training on GPU $GPU with num_symbols=$ns"
    CUDA_VISIBLE_DEVICES=$GPU python3 udeepsc_main.py \
        --model  UDeepSC_NOMANoSIC_model  \
        --output_dir output/bandwidth/sb$ns  \
        --batch_size 50 \
        --input_size 32 \
        --lr  3e-5 \
        --epochs 150  \
        --opt_betas 0.95 0.99  \
        --save_freq 50   \
        --ta_perform msa \
        --log_interval 10 \
        --seed 1000 \
        --num_symbols $ns \

done

echo "All training runs complete."