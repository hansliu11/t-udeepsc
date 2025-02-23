CUDA_VISIBLE_DEVICES=0  python3  udeepsc_main.py \
    --model  UDeepSC_new_model  \
    --output_dir output  \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 50   \
    --ta_perform msa \
    --log_interval 10 \
    # --model UDeepSC_NOMA_model \
    # --device cpu
    # --resume ckpt_record_12dB_single/ckpt_msa/checkpoint-203.pth\
    # --eval

   

   
 
 
  
  