CUDA_VISIBLE_DEVICES=0 python3  tdeepsc_main.py \
    --model  TDeepSC_ave_model  \
    --output_dir output  \
    --batch_size 64\
    --input_size 32 \
    --lr  3e-5 \
    --epochs 150  \
    --opt_betas 0.95 0.99  \
    --save_freq 50   \
    --ta_perform ave \
  
   