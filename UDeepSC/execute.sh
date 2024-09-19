CUDA_VISIBLE_DEVICES=3  python3  udeepsc_main.py \
    --model  UDeepSC_new_model  \
    --output_dir output   \
    --batch_size 16 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 500  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform imgr \
    # --resume ckpt_record_12dB_single/ckpt_msa/checkpoint-203.pth\
    # --eval
   
  

   

   
 
 
  
  