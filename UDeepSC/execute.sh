# CUDA_VISIBLE_DEVICES=0  
python3  udeepsc_main.py \
    --model  UDeepSC_new_model  \
    --output_dir output   \
    --batch_size 32 \
    --input_size 32 \
    --lr  1e-5 \
    --epochs 150  \
    --opt_betas 0.95 0.99  \
    --save_freq 50   \
    --ta_perform textr \
    --log_interval 10 \
    # --device cpu
    # --resume ckpt_record_12dB_single/ckpt_msa/checkpoint-203.pth\
    # --eval

   

   
 
 
  
  