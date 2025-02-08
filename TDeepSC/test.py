import torch
from pathlib import Path
from loguru import logger
import re
from transformers import BertTokenizer
from tqdm import tqdm
import datetime

import model  
from utils import *
from base_args import get_args
from datasets import build_dataset, collate_fn
from torch.utils.data import Subset
from vqa_utils import VQA_Tool, VQA_Eval

def save_result_JSON(results: list, output:str):
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    
    now = now.strftime('%m%d-%H%M')
    output = "result/" + output + now + ".json"
    logger.info(f"Save result to {output}...")
    
    # Save the list to a JSON file
    with open(output, "w") as file:
        json.dump(results, file)

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

def get_test_samples(args, batch_size=5):
    """Get first "batch size" of test sample data in dataloader"""
    testset = build_dataset(is_train=False, args=args)
    sampler_test = torch.utils.data.SequentialSampler(testset)
    test_dataloader= torch.utils.data.DataLoader(
        testset, sampler=sampler_test, batch_size=int(1.0 * batch_size),
        num_workers=4, pin_memory=args.pin_mem, drop_last=False)
    
    inputs, targets = next(iter(test_dataloader))
    # print(str_type(inputs))
    return inputs, targets

def get_test_dataloader(args, batch_size=5):
    """Return testset and test dataloader"""
    testset = build_dataset(is_train=False, args=args)

    sampler_test = torch.utils.data.SequentialSampler(testset)
    Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None 
    test_dataloader= torch.utils.data.DataLoader(
        testset, sampler=sampler_test, batch_size=int(1.0 * args.batch_size),
        num_workers=4, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    
    return testset, test_dataloader

def test_SNR(ta_perform:str, SNRrange:list[int], model_path, args,device, dataloader:Iterable):
    """
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
    model.to(device)
    
    model.eval()
    
    if ta_perform.startswith('vqa'):
        dataset = dataloader.dataset
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        
        ans_ix_list = []
        i = 0
        snr = torch.FloatTensor([snr])
        
        for (imgs, texts, targets) in tqdm(dataloader):
            imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
            batch_size = imgs.shape[0]
            i += batch_size  
            outputs = model(img=imgs, text=texts, ta_perform=ta_perform, test_snr=snr)
            pred_np = outputs.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)
            if pred_argmax.shape[0] != dataset.configs.eval_batch_size:
                pred_argmax = np.pad(
                    pred_argmax,(0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
                    mode='constant',constant_values=-1)
            ans_ix_list.append(pred_argmax)
            
        ans_ix_list = np.array(ans_ix_list).reshape(-1)
        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])}for qix in range(qid_list.__len__())]

        result_eval_file = 'vqaeval_result/result_run_' + dataset.configs.version + '.json'
        print('Save the result to file: {}'.format(result_eval_file))
        json.dump(result, open(result_eval_file, 'w'))

        # create vqa object and vqaRes object
        ques_file_path = dataset.configs.question_path['val']
        ans_file_path = dataset.configs.answer_path['val']
        vqa = VQA_Tool(ans_file_path, ques_file_path)
        vqaRes = vqa.loadRes(result_eval_file, ques_file_path)
        vqaEval = VQA_Eval(vqa, vqaRes, n=2)  
        vqaEval.evaluate()
        states = vqaEval.accuracy
        
        return states['overall']
    
    elif ta_perform.startswith('msa'):
        snr = torch.FloatTensor([snr])
        logger.info(f"Test SNR = {snr}")
        
        y_true, y_pred = [], []
        for (imgs, texts, speechs, targets) in tqdm(dataloader):
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
            outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, test_snr=snr)
            y_pred.append(outputs.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
    
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        acc = calc_metrics(y_true, y_pred)       
        
        return acc * 100
    
def main_test_SNR_single():
    opts = get_args()
    ta_perform = 'vqa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    # power_constraint = [0.5, 1, 1.5]
    power_constraint = [0.5, 1.5]
    result_output = ta_perform + "_result"
    
    chart_args = {
        'channel_type' : "AWGN channel",
        'output': "acc_" + ta_perform,
        'y_axis': "Accuracy (%)",
        # "y_lim" : [55, 81, 5],
        "y_lim" : [20, 62, 10],
    }
    
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
        task_fold = 'ckpt_vqa'
    elif ta_perform.startswith('msa'):
        task_fold = 'ckpt_msa'

    folder = Path('./output'+ '/' + task_fold)
    
    # udeepsc
    best_model_path = get_best_checkpoint(folder, "checkpoint")
    print(f'{best_model_path = }')
    
    
    opts.model = f'TDeepSC_{ta_perform}_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 32
    
    testset, dataloader = get_test_dataloader(opts)
    SNRrange = [-6, 12]
    
    metric1 = test_SNR(ta_perform, SNRrange, best_model_path, opts, device, dataloader)
    
    # print(snr12_bleus)
    
    x = [i for i in range(SNRrange[0], SNRrange[1] + 1)]
    models = [metric1] * len(x)
    
    save_result_JSON(models, result_output)

if __name__ == '__main__':
    main_test_SNR_single()