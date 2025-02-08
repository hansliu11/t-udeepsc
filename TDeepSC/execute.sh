CUDA_VISIBLE_DEVICES=0 python3  tdeepsc_main.py \
    --model  TDeepSC_vqa_model  \
    --output_dir output  \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 50   \
    --ta_perform vqa \
  
   