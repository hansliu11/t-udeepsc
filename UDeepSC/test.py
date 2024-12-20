import torch
import math
import torch.nn as nn
from pathlib import Path
from loguru import logger
import re
from pytorch_transformers import BertTokenizer

import model  
from utils import *
from base_args import get_args
from timm.utils import AverageMeter
from einops import rearrange
from datasets import build_dataset_test

def get_best_checkpoint(folder: Path, label="checkpoint"):
    """
        get the biggest checkpoint-{number}*
    """
    
    pattern = re.compile(label + '(_|-)(\d+)*')
    
    paths = {}
    for p in folder.glob('*'):
        m = re.match(pattern, p.name)
        
        if m is None:
            continue
        paths[int(m.group(2))] = p
    
    # ep, path = max(paths.items(), key=lambda a : a[0])
    path = paths[max(paths)]

    return path       

def get_sample_text(args, batch_size=5):
    testset = build_dataset_test(is_train=False, args=args)
    sampler_test = torch.utils.data.SequentialSampler(testset)
    test_dataloader= torch.utils.data.DataLoader(
        testset, sampler=sampler_test, batch_size=int(1.0 * batch_size),
        num_workers=4, pin_memory=args.pin_mem, drop_last=False)
    
    inputs, targets = next(iter(test_dataloader))
    print(str_type(inputs))
    return inputs, targets

def text_test(ta_perform:str, texts:list[str], snr:torch.FloatTensor, model, device):
    logger.info("Start test textr")
    
    batch_size = texts.size(0)
    bleu_meter = AverageMeter()
    
    texts = texts.to(device)
    model.eval() ## set model test mode
    outputs = model(text=texts, ta_perform=ta_perform, test_snr=snr)
    
    texts = texts[:,1:]
    preds = torch.zeros_like(texts)
    for i in range(outputs.shape[1]):
        preds[:,i] = outputs[:,i].max(-1)[-1] 
    
    preds = tokens2sentence(preds)
    texts = tokens2sentence(texts)
    bleu_meter.update(computebleu(preds, texts)/batch_size, n=batch_size)
    print(f'Test at SNR = {snr.item()} db')
    for i, (pred, target) in enumerate(zip(preds, texts)):
        if i % 5 == 0:
            print(f'Sentence {i + 1}')
            print('Transmitted: ' + ' '.join(target[:-1]))
            print('Recovered: ' + ' '.join(pred[:-1]))
            print(f'Bleu Score: {bleu_meter.avg:.3f}')
    
    return preds

def text_test_single(ta_perform:str, text, snr:torch.FloatTensor, model, device):
    logger.info("Start test textr")
    
    texts = texts.to(device)
    model.eval()
    outputs = model(text=texts, ta_perform=ta_perform, test_snr=snr)
    
    text = text[1:]
    preds = torch.zeros_like(text)
    for i in range(outputs.shape):
        preds[i] = outputs[i].max(-1)[-1] 
    
    preds = tokens2sentence(preds)
    texts = tokens2sentence(texts)
    for i, (pred, target) in enumerate(zip(preds, texts)):
        print(f'Sentence {i + 1}')
        print('Transmitted: ' + ' '.join(target[:-1]))
        print('Recovered: ' + ' '.join(pred[:-1]))
    
    return preds

def test_SNR(ta_perform:str, SNRrange:list[int], model_path, args,device, dataset):
    """
    TODO
    Test model on different SNR
    Inputs:
        SNRrange: Test SNR range which is a list with length 2
                    (should be [min SNR,  max SNR])
    """
    logger.info("Start test different SNR")
    
    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
    model.eval()
    if ta_perform.startswith('imgr'):
        imgs, targets = dataset
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        psnr_meter = AverageMeter()
        psnr_list = []
        
        for i in range(SNRrange[0], SNRrange[1] + 2, 2):
            snr = torch.FloatTensor([i])
            logger.info(f"Test SNR = {snr}")
            outputs = model(img=imgs, ta_perform=ta_perform, test_SNR=snr)
            outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
        
            predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
            targets = torch.chunk(imgs, chunks=imgs.size(0), dim=0)
            psnr_vals = calc_psnr(predictions, targets)
            psnr_list.extend(psnr_vals)
            psnr_meter.update(torch.mean(torch.tensor(psnr_vals)).item(), n=batch_size)
            
            psnr_list.append(psnr_meter.avg)
        
        return psnr_list
    
    elif ta_perform.startswith('textr'):
        texts, _ = dataset
        batch_size = texts.size(0)
        avg_bleu = []
        bleu_meter = AverageMeter()
        for i in range(SNRrange[0], SNRrange[1]):
            snr = torch.FloatTensor([i])
            logger.info(f"Test SNR = {snr}")
            outputs = model(text=texts, ta_perform=ta_perform, test_snr=snr)
        
            targets = texts[:,1:]
            preds = torch.zeros_like(targets)
            for i in range(outputs.shape[1]):
                preds[:,i] = outputs[:,i].max(-1)[-1] 
            
            preds = tokens2sentence(preds)
            targets = tokens2sentence(targets)
            bleu_meter.update(computebleu(preds, targets)/batch_size, n=batch_size)
            
            avg_bleu.append(bleu_meter.avg)
        
        return avg_bleu
    
def test_features(ta:str, test_snr: torch.FloatTensor, model_path, args,device, dataset):
    logger.info("Start test features before transmission and after")
    
    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
    # TODO Need modify
    if ta.startswith('img'):
        imgs = dataset[0].to(device, non_blocking=True)
        targets = dataset[1].to(device, non_blocking=True)
    elif ta.startswith('text'):
        texts = dataset[0].to(device, non_blocking=True)
        targets = dataset[1].to(device, non_blocking=True)
    elif ta.startswith('vqa'):
        imgs = dataset[0].to(device, non_blocking=True)
        texts = dataset[1].to(device, non_blocking=True)
        targets = dataset[2].to(device, non_blocking=True)
    elif ta.startswith('msa'):
        imgs = dataset[0].to(device, non_blocking=True)
        texts = dataset[1].to(device, non_blocking=True) 
        speechs = dataset[2].to(device, non_blocking=True)
        targets = dataset[3].to(device, non_blocking=True)
    else:
        raise NotImplementedError()
    
    model.eval()
    signals = model.get_signals(text=texts, ta_perform=ta, SNRdb=test_snr)
    text_signals_before, text_signals_after = signals['text']
    batch_size = text_signals_before.shape[0]
    for i in range(batch_size):
        print("Before transmitting: ")
        print(str_type(text_signals_before[i]))
        print(text_signals_before[i])
        
        print("After transmitting: ")
        print(str_type(text_signals_after[i]))
        print(text_signals_after[i])
        
    return text_signals_before, text_signals_after

def main_test_signals():
    opts = get_args()
    ta_perform = 'textr'
    device = 'cpu'
    
    if ta_perform.startswith('imgc'):
        task_fold = 'imgc'
    elif ta_perform.startswith('imgr'):
        task_fold = 'imgr'
    elif ta_perform.startswith('textc'):
        task_fold = 'textc'
    elif ta_perform.startswith('textr'):
        task_fold = 'ckpt_textr'
        task_fold = 'textr_smooth_005'
    elif ta_perform.startswith('vqa'):
        task_fold = 'vqa'
    elif ta_perform.startswith('msa'):
        task_fold = 'msa'

    folder = Path('./output'+ '/' + task_fold)
    
    best_model_path1 = get_best_checkpoint(folder, "snr12")
    opts.model = 'UDeepSC_new_model'
    opts.ta_perform = ta_perform
    
    texts, targets = get_sample_text(opts)
    test_snr = torch.FloatTensor([12])
    
    signals = test_features(ta_perform, test_snr, best_model_path1, opts, device, [texts, targets])

def main_test_textr_SNR():
    opts = get_args()
    ta_perform = 'textr'
    device = 'cpu'
    
    if ta_perform.startswith('imgc'):
        task_fold = 'imgc'
    elif ta_perform.startswith('imgr'):
        task_fold = 'imgr'
    elif ta_perform.startswith('textc'):
        task_fold = 'textc'
    elif ta_perform.startswith('textr'):
        task_fold = 'ckpt_textr'
        task_fold = 'textr_smooth_01'
    elif ta_perform.startswith('vqa'):
        task_fold = 'vqa'
    elif ta_perform.startswith('msa'):
        task_fold = 'msa'

    folder = Path('./output'+ '/' + task_fold)
    
    # test model trained on snr 12
    best_model_path1 = get_best_checkpoint(folder, "snr12")
    print(f'{best_model_path1 = }')
    
    # test model trained on snr -2
    best_model_path2 = get_best_checkpoint(folder, "snr-2")
    print(f'{best_model_path2 = }')
    
    opts.model = 'UDeepSC_new_model'
    opts.ta_perform = ta_perform
    
    texts, targets = get_sample_text(opts, batch_size=32)
    SNRrange = [-6, 12]
    
    snr12_bleus = test_SNR(ta_perform, SNRrange, best_model_path1, opts, device, [texts, targets])
    
    snrNeg_bleus = test_SNR(ta_perform, SNRrange, best_model_path2, opts, device, [texts, targets])
    
    # print(snr12_bleus)
    
    x = [i for i in range(SNRrange[0], SNRrange[1] + 2, 2)]
    models = [snr12_bleus, snrNeg_bleus]
    labels = ["Textr-LSCE (SNR = 12)", "Textr-LSCE (SNR = -2)"]
    draw_line_chart(x, models, labels, "AWGN","SNR/db", "Bleu score", output="bleu_SNR")

def main_test1():
    opts = get_args()
    ta_perform = 'textr'
    device = 'cpu'
    
    if ta_perform.startswith('imgc'):
        task_fold = 'imgc'
    elif ta_perform.startswith('imgr'):
        task_fold = 'imgr'
    elif ta_perform.startswith('textc'):
        task_fold = 'textc'
    elif ta_perform.startswith('textr'):
        task_fold = 'ckpt_textr'
        task_fold = 'textr_smooth_01'
    elif ta_perform.startswith('vqa'):
        task_fold = 'vqa'
    elif ta_perform.startswith('msa'):
        task_fold = 'msa'

    folder = Path('./output'+ '/' + task_fold)
    best_model_path = get_best_checkpoint(folder, 'snr12')
    print(f'{best_model_path = }')
    opts.model = 'UDeepSC_new_model'
    opts.resume = best_model_path
    opts.ta_perform = ta_perform
    
    model = get_model(opts)
    print(f'{opts.resume = }')
    checkpoint_model = load_checkpoint(model, opts)
    load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    
    # text = "Jack lives in HsinChu and he is 25 years old"
    
    inputs, _ = get_sample_text(opts, batch_size=32)
    
    # SNR for testing, default = 12
    # range: [-6, 12]
    test_snr = torch.FloatTensor([12])
    
    received = text_test(ta_perform, inputs, test_snr, model, device)

if __name__ == '__main__':
    # main_test1()
    main_test_textr_SNR()
    # main_test_signals()
