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
from datasets import build_dataset_test

def get_best_checkpoint(folder: Path):
    """
        get the biggest checkpoint-{number}*
    """
    
    pattern = re.compile('checkpoint-(\d+)*')
    
    paths = {}
    for p in folder.glob('*'):
        m = re.match(pattern, p.name)
        
        if m is None:
            continue
        paths[int(m.group(1))] = p
    
    # ep, path = max(paths.items(), key=lambda a : a[0])
    path = paths[max(paths)]

    return path       

def text_test(ta_peform:str, texts:list[str], path, args, device):
    logger.info("Start test textr")
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
    texts = texts.to(device)
    outputs = model(text=texts, ta_perform=ta_perform)
    
    texts = texts[:,1:]
    preds = torch.zeros_like(texts)
    for i in range(outputs.shape[1]):
        preds[:,i] = outputs[:,i].max(-1)[-1] 
    
    preds = tokens2sentence(preds)
    texts = tokens2sentence(texts)
    for i, (pred, target) in enumerate(zip(preds, texts)):
        print(f'Sentence {i + 1}')
        print('Transmitted: ' + ' '.join(target[:-1]))
        print('Recovered: ' + ' '.join(pred[:-1]))
    
    return preds
    

if __name__ == '__main__':
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
    elif ta_perform.startswith('vqa'):
        task_fold = 'vqa'
    elif ta_perform.startswith('msa'):
        task_fold = 'msa'

    folder = Path('./output'+ '/' + task_fold)
    best_model_path = get_best_checkpoint(folder)
    print(f'{best_model_path = }')
    opts.model = 'UDeepSC_new_model'
    opts.resume = best_model_path
    opts.ta_perform = ta_perform
    
    # text = "Jack lives in HsinChu and he is 25 years old"
    
    def get_sample_text():
        batch_size = 5
        testset = build_dataset_test(is_train=False, args=opts)
        sampler_test = torch.utils.data.SequentialSampler(testset)
        test_dataloader= torch.utils.data.DataLoader(
            testset, sampler=sampler_test, batch_size=int(1.0 * batch_size),
            num_workers=4, pin_memory=opts.pin_mem, drop_last=False)
        
        inputs, _ = next(iter(test_dataloader))
        print(str_type(inputs))
        return inputs
    
    inputs = get_sample_text()
    
    received = text_test(ta_perform, inputs, best_model_path, opts, device)
    