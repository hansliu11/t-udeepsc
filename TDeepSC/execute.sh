CUDA_VISIBLE_DEVICES=0 python3  tdeepsc_main.py \
    --model  TDeepSC_msa_model  \
    --output_dir output  \
    --batch_size 64\
    --input_size 32 \
    --lr  1e-5 \
    --epochs 250  \
    --opt_betas 0.95 0.99  \
    --save_freq 50   \
    --ta_perform msa \
  
   