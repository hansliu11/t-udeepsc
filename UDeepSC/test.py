import torch
from pathlib import Path
from loguru import logger
import re
from transformers import BertTokenizer
from tqdm import tqdm
import json
import datetime
from timeit import default_timer

import model  
from utils import *
from channel import signal_power
from base_args import get_args
from timm.utils import AverageMeter
from einops import rearrange
from datasets import build_dataset_test, collate_fn, collate_fn_Shuff
from torch.utils.data import Subset
import matplotlib.pyplot as plt
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
    
    pattern = re.compile(label + '(-)(\d+)*')
    
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
    testset = build_dataset_test(is_train=False, args=args)
    sampler_test = torch.utils.data.SequentialSampler(testset)
    Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None 
    test_dataloader= torch.utils.data.DataLoader(
        testset, sampler=sampler_test, batch_size=int(1.0 * batch_size),
        num_workers=4, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    
    imgs, texts, speechs, targets = None, None, None, None
    if args.ta_perform.startswith('img'):
        imgs, targets = next(iter(test_dataloader))
    elif args.ta_perform.startswith('text'):
        texts, targets = next(iter(test_dataloader))
    elif args.ta_perform.startswith('vqa'):
        imgs, texts, targets = next(iter(test_dataloader))
    elif args.ta_perform.startswith('msa'):
        imgs, texts, speechs, targets = next(iter(test_dataloader))
        
    else:
        raise NotImplementedError()

    sel_batch = [imgs, texts, speechs] 
    print(str_type(sel_batch))
    return sel_batch, targets

def get_test_dataloader(args, shuffle=False, infra=False, shifts=0):
    """Return testset and test dataloader"""
    split = 'test' if args.ta_perform == 'ave' else 'val'
    testset = build_dataset_test(is_train=False, args=args, split=split, infra=infra, shuffle=shuffle, shifts=shifts)
    
    if args.textr_euro:
        indices = list(range(args.batch_size * 1000))
        testset = Subset(testset, indices)

    sampler_test = torch.utils.data.SequentialSampler(testset)
    Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None 
    test_dataloader= torch.utils.data.DataLoader(
        testset, sampler=sampler_test, batch_size=int(1.0 * args.batch_size),
        num_workers=4, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    
    return testset, test_dataloader

def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)

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
    
    preds = tokens2sentenceV2(preds)
    texts = tokens2sentenceV2(texts)
    bleu_meter.update(computebleu(preds, texts)/batch_size, n=batch_size)
    print(f'Test at SNR = {snr.item()} db')
    print(f'Avg Bleu Score: {bleu_meter.avg:.3f}')
    for i, (pred, target) in enumerate(zip(preds, texts)):
        if i < 5:
            print(f'Sentence {i + 1}')
            print('Transmitted: ' + ' '.join(target[:-1]))
            print('Recovered: ' + ' '.join(pred[:-1]))
    
    return preds

def text_test_single(ta_perform:str, text, snr:torch.FloatTensor, model, device):
    logger.info("Start test textr")
    
    text = text.to(device)
    model.eval()
    outputs = model(text=text, ta_perform=ta_perform, test_snr=snr)
    
    text = text[:,1:]
    pred = torch.zeros_like(text)
    for i in range(outputs.shape[1]):
        pred[:, i] = outputs[:,i].max(-1)[-1] 
    
    pred = tokens2sentenceV2(pred)
    text = tokens2sentenceV2(text)
    bleu_score = computebleu(pred, text)
    
    print(f'Test at SNR = {snr.item()} db')
    # for i, (pred, target) in enumerate(zip(preds, text)):
    print('Transmitted: ' + ' '.join(text[0][:-1]))
    print('Recovered: ' + ' '.join(pred[0][:-1]))
    print(f'Bleu Score: {bleu_score:.3f}')
    
    return pred

def text_test_BLEU(ta_perform:str, textLoader: Iterable, snr:torch.FloatTensor, model, device):
    logger.info("Start test textr")
    
    bleu_meter = AverageMeter()
    result = [] ## bleu score of each sentence
    
    model.eval() ## set model test mode
    print(f'Test at SNR = {snr.item()} db')
    for (texts, targets) in tqdm(textLoader):
        texts, targets = texts.to(device), targets.to(device)
        batch_size = targets.size(0)
        
        outputs = model(text=texts, ta_perform=ta_perform, test_snr=snr)
        
        targets = targets[:,1:]
        preds = torch.zeros_like(targets)
        for i in range(outputs.shape[1]):
            preds[:,i] = outputs[:,i].max(-1)[-1] 
        
        preds = tokens2sentence(preds)
        targets = tokens2sentence(targets)
        
        for pred, target in zip(preds, targets):
            # print(f'Pred: {len(pred)}')
            # print(f'Tar: {len(target)}')
            result.append(computebleu([pred], [target]))
        
        bleu_meter.update(computebleu(preds, targets)/batch_size, n=batch_size)
            
    test_stat = {'bleu': result, 'average': bleu_meter.avg}  
    
    return test_stat


def test_SNR(ta_perform:str, SNRrange:list[int], power_constraint, model_path, args,device, dataloader:Iterable):
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
    if ta_perform.startswith('imgc'):    
        avg_acc = []
        for snr in range(SNRrange[0], SNRrange[1] + 1):
            acc_meter = AverageMeter()
            snr = torch.FloatTensor([snr])  
            logger.info(f"Test SNR = {snr}")
            for (imgs, targets) in tqdm(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(img=imgs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                batch_size = targets.size(0)
                idx, predicted = outputs.max(1)
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
                
            avg_acc.append(acc_meter.avg)
        
        return avg_acc
    
    elif ta_perform.startswith('textr'):   
        avg_bleu = []
        for snr in range(SNRrange[0], SNRrange[1] + 1):
            bleu_meter = AverageMeter()
            snr = torch.FloatTensor([snr])  
            logger.info(f"Test SNR = {snr}")
            for (texts, targets) in tqdm(dataloader):
                batch_size = texts.size(0)
                texts, targets = texts.to(device), targets.to(device)
                
                
                outputs = model(text=texts, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
            
                targets = texts[:,1:]
                preds = torch.zeros_like(targets)
                for i in range(outputs.shape[1]):
                    preds[:,i] = outputs[:,i].max(-1)[-1] 
                
                preds = tokens2sentence(preds)
                targets = tokens2sentence(targets) 
                
                # bleu = computebleu(preds, targets) / batch_size
                bleu_meter.update(computebleu(preds, targets)/batch_size, n=batch_size)
                
            avg_bleu.append(bleu_meter.avg)
        
        return avg_bleu
    
    elif ta_perform.startswith('vqa'):
        dataset = dataloader.dataset
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        avg_acc = []
        for snr in range(SNRrange[0], SNRrange[1] + 1):
            ans_ix_list = []
            i = 0
            snr = torch.FloatTensor([snr])
            logger.info(f"Test SNR = {snr}")
            
            for (imgs, texts, targets) in tqdm(dataloader):
                imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
                batch_size = imgs.shape[0]
                i += batch_size  
                outputs = model(img=imgs, text=texts, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
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
            avg_acc.append(states['overall'])
        
        return avg_acc
    
    elif ta_perform.startswith('msa'):
        avg_acc = []
        for snr in range(SNRrange[0], SNRrange[1] + 1):
            snr = torch.FloatTensor([snr])
            logger.info(f"Test SNR = {snr}")
            
            y_true, y_pred = [], []
            for (imgs, texts, speechs, targets) in tqdm(dataloader):
                imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
                outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                # outputs = model(img=imgs, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
        
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            acc = calc_metrics(y_true, y_pred) 
            avg_acc.append(acc * 100)       
        
        return avg_acc
    
    elif ta_perform.startswith('ave'):
        avg_acc = []
        nb_batch = len(dataloader)
        for snr in range(SNRrange[0], SNRrange[1] + 1):
            snr = torch.FloatTensor([snr])
            logger.info(f"Test SNR = {snr}")
            
            y_true, y_pred = [], []
            for (imgs, speechs, targets) in tqdm(dataloader):
                imgs, speechs, targets = imgs.to(device), speechs.to(device), targets.to(device)
                # img2s = img2s.to(device) 
                outputs = model(img=imgs, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                # outputs = model(img=imgs, speech=speechs, img2=img2s, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
        
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            acc = compute_acc_AVE(y_true, y_pred, nb_batch) 
            avg_acc.append(acc * 100)       
        
        return avg_acc
        
def test_shift_data(ta_perform:str, shift_steps:list[int], power_constraint, model_path, args, device, snr=0):
    logger.info("Start test shift dataset")
    
    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    
    snr = torch.FloatTensor([snr])

    model.eval()
    progress_bar = tqdm(shift_steps, leave=False, total=len(shift_steps), dynamic_ncols=True)
    if ta_perform.startswith('msa'):
        avg_acc = []
        for step in progress_bar:
            # collate = CollateShuff(step)
            testset, dataloader = get_test_dataloader(args, shifts=step)
            
            test_progress = tqdm(dataloader, leave=False, total=len(dataloader), dynamic_ncols=True)
            y_true, y_pred = [], []
            for (imgs, texts, speechs, targets) in test_progress:
                imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
                outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
        
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            acc = calc_metrics(y_true, y_pred) 
            avg_acc.append(acc * 100)       
        
        return avg_acc

def test_features(ta:str, test_snr: torch.FloatTensor, power_constraint, model_path, args, device, sel_batch):
    logger.info("Start test features before transmission and after")
    
    imgs, texts, speechs = sel_batch

    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
    # TODO Need modify
    model.eval()
    
    if ta.startswith('img'):
        Tx_sigs, Rx_sigs = model.get_signals(img=imgs, ta_perform=ta, test_snr=test_snr)
    elif ta.startswith('text'):
        Tx_sigs, Rx_sigs = model.get_signals(text=texts, ta_perform=ta, test_snr=test_snr)
    elif ta.startswith('vqa'):
        Tx_sigs, Rx_sigs = model.get_signals(img=imgs, text=texts, ta_perform=ta, power_constraint=power_constraint, test_snr=test_snr)
    elif ta.startswith('msa'):
        Tx_sigs, Rx_sigs = model.get_signals(img=imgs, text=texts, speech=speechs, ta_perform=ta, power_constraint=power_constraint, test_snr=test_snr)
    else:
        raise NotImplementedError()
    
    batch_size = Tx_sigs[0].shape[0]

    for i in range(batch_size):
        for tx, rx in zip(Tx_sigs, Rx_sigs):
            print(f"Before transmitting {i}: ")
            print(str_type(tx[i]))
            print(tx[i])
            
            print(f"After transmitting {i}: ")
            print(str_type(rx[i]))
            print(rx[i])

            ## pause
            input("press any key..")
        
    return Tx_sigs, Rx_sigs

def test_power_norm(ta_perform:str, power_constraint, model_path, args,device, dataloader:Iterable, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    
    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    
    model.eval()
    impose_norms = []
    norm_imposes = []
    if ta_perform.startswith('msa'):
        snr = torch.FloatTensor([0])
        for (imgs, texts, speechs, targets) in tqdm(dataloader):
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
            imposed_norm_sig, norm_imposed_sig  = model.get_signals(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint, test_snr=snr)

            impose_norms.append(imposed_norm_sig)
            norm_imposes.append(norm_imposed_sig)
    
    with open(output_path / 'signal_pow.txt', 'w') as fout:
        for x, x_norm in zip(impose_norms, norm_imposes):
            print(f"{str_type(x)=}", file=fout)
            print(f"{str_type(x_norm)=}", file=fout)
            print(f"Power normalization after stack: \n {signal_power(x.flatten(2))=}", file=fout)
            print(f"Power normalization before stack: \n {signal_power(x_norm.flatten(2))=}", file=fout)


def main_test_signals():
    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:0'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    # power_constraint = [0.5, 1, 1.5]
    # power_constraint = [0.5, 1.5]
    # power_constraint = [0.5]
    root = './output'
    models_dir = Path(root)

    print(f"Power Constraint: {power_constraint_static}")
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'
    
    # udeepsc
    best_model_path = get_best_checkpoint(folder_noSIC, "powerSum")
    print(f'{best_model_path = }')
    
    
    opts.model = 'UDeepSC_NOMANoSIC_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 16
    
    testset, dataloader = get_test_dataloader(opts)
    
    test_power_norm(ta_perform, power_constraint_static, best_model_path, opts, device, dataloader, Path("/home/ldap/hansliu/t-udeepsc/UDeepSC/tmp/20250419_msa_noSIC_test"))

def main_test_SNR():
    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [0.5, 1, 1.5]
    # power_constraint = [1.6, 0.9, 0.4]
    # power_constraint = [0.5, 1.5]
    result_output = ta_perform + "_result" + "(snr-2)"
    
    chart_args = {
        'channel_type' : "AWGN channel",
        'output': "acc_" + ta_perform + "_result" + "(snr-2)",
        'y_axis': "Accuracy (%)",
        "y_lim" : [55, 82, 5],
        # "y_lim" : [20, 62, 10],
        "vqa_upper": 55.8,
        "msa_upper": 83
    }
    
    root = './output'
    models_dir = Path(root)
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'

    opts.ta_perform = ta_perform
    opts.batch_size = 32
    
    testset, dataloader = get_test_dataloader(opts, shuffle=False, infra=False)
    SNRrange = [-6, 12]

    udeepsc_res = []
    perfectSIC_res = []
    NOMASIC_res = []
    NOMAnoSIC_res = []

    test_start = default_timer()

    for idx in range(1, 6):
        for _ in range(10):
            # udeepsc
            best_model_path1 = get_best_checkpoint(folder, f"M3lowSNR_{idx}")
            print(f'{best_model_path1 = }')

            # udeepsc Noma perfect SIC
            best_model_path4 = get_best_checkpoint(folder_pfSIC, f"M3lowSNR_{idx}")
            print(f'{best_model_path4 = }')
            
            opts.model = 'UDeepSC_SepCD_model'
            metric1 = test_SNR(ta_perform, SNRrange, power_constraint_static, best_model_path1, opts, device, dataloader)
            udeepsc_res.append(metric1)

            metric4 = test_SNR(ta_perform, SNRrange, power_constraint, best_model_path4, opts, device, dataloader)
            perfectSIC_res.append(metric4)
            
            # udeepsc Noma
            best_model_path2 = get_best_checkpoint(folderSIC, f"lowSNR_{idx}")
            print(f'{best_model_path2 = }')

            opts.model = 'UDeepSC_NOMA_new_model'
            metric2 = test_SNR(ta_perform, SNRrange, power_constraint, best_model_path2, opts, device, dataloader)
            NOMASIC_res.append(metric2)

            # udeepsc Noma no SIC
            best_model_path3 = get_best_checkpoint(folder_noSIC, f"lowSNR_{idx}")
            print(f'{best_model_path3 = }')
            
            # power_constraint_static = [3, 3, 3]
            opts.model = 'UDeepSC_NOMANoSIC_model'
            metric3 = test_SNR(ta_perform, SNRrange, power_constraint_static, best_model_path3, opts, device, dataloader)
            NOMAnoSIC_res.append(metric3)     
    
    udeepsc_res = np.array(udeepsc_res).mean(axis=0).tolist()
    perfectSIC_res = np.array(perfectSIC_res).mean(axis=0).tolist()
    NOMASIC_res = np.array(NOMASIC_res).mean(axis=0).tolist()
    NOMAnoSIC_res = np.array(NOMAnoSIC_res).mean(axis=0).tolist()
    
    x = [i for i in range(SNRrange[0], SNRrange[1] + 1)]
    # upper_bound = [chart_args[f'{ta_perform}_upper']] * len(x)
    models = [udeepsc_res, perfectSIC_res, NOMASIC_res, NOMAnoSIC_res]
    test_time = default_timer() - test_start
    
    test_set = {
        "Title": "MSA test, trained on SNR -2",
        # 'Title': "Ave test",
        "time": str(datetime.timedelta(seconds=test_time)),
        "Test Samples": len(testset),
        "power": power_constraint, 
        "noSIC power": power_constraint_static,
        'udeepsc': str(best_model_path1),
        'perfect SIC': str(best_model_path4),
        'udeepsc NOMA': str(best_model_path2),
        'no SIC': str(best_model_path3),
        'result': models
    }
    save_result_JSON(test_set, result_output)
    
    labels = ["U-DeepSC", "U-DeepSC with perfect SIC", "U-DeepSC NOMA (with SIC)", "U-DeepSC NOMA (w/o SIC)"]
    draw_line_chart(x, models, 
                    y_lim= chart_args['y_lim'],
                    labels=labels, 
                    title=chart_args['channel_type'], 
                    xlabel="SNR/dB", 
                    ylabel=chart_args['y_axis'], 
                    output=chart_args['output']
                    )

def main_test_SNR_single():
    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [0.5, 1, 1.5]
    # power_constraint = [0.5, 1.5]
    # power_constraint = [0.5]
    result_output = ta_perform + "_result_single"
    root = './output'
    models_dir = Path(root)

    print(f"Power Constraint: {power_constraint}")
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'
    
    # udeepsc
    best_model_path = get_best_checkpoint(folder_noSIC, "power111_5")
    # best_model_path = folder_noSIC / "checkpoint-6.pth"
    print(f'{best_model_path = }')
    
    
    opts.model = 'UDeepSC_NOMANoSIC_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 16
    
    testset, dataloader = get_test_dataloader(opts, infra=False)
    SNRrange = [-6, 12]
    
    metric1 = test_SNR(ta_perform, SNRrange, power_constraint_static, best_model_path, opts, device, dataloader)
    
    
    models = [metric1]
    test_setting = {
        # "Title": "Msa test image & speech",
        "Title": "Msa test",
        "Test Samples": len(testset),
        "power": power_constraint, 
        "noSIC power": power_constraint_static,
        "Model": str(best_model_path),
        "Result": models
    }
    
    save_result_JSON(test_setting, result_output)
    
def main_test_Modal_SNR():
    opts = get_args()
    ta_perform = 'ave'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [1.5, 0.5]
    # power_constraint = [1.6, 0.4]
    # power_constraint = [1]
    result_output = ta_perform + "_result" + "_Homo"
    
    chart_args = {
        'channel_type' : "AWGN channel",
        'output': "acc_" + ta_perform + "_Homo",
        'y_axis': "Accuracy (%)",
        "y_lim" : [55, 82, 5],
        # "y_lim" : [20, 62, 10],
        "vqa_upper": 55.8,
        "msa_upper": 83
    }
    
    root = './output'
    models_dir = Path(root)
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'
    
    # udeepsc
    best_model_path1 = get_best_checkpoint(folder, "checkpoint")
    print(f'{best_model_path1 = }')
    
    # udeepsc Noma
    best_model_path2 = get_best_checkpoint(folderSIC, "checkpoint")
    print(f'{best_model_path2 = }')
    
    # udeepsc Noma no SIC
    best_model_path3 = get_best_checkpoint(folder_noSIC, "checkpoint")
    print(f'{best_model_path3 = }')

    # udeepsc Noma perfect SIC
    best_model_path4 = get_best_checkpoint(folder_pfSIC, "checkpoint")
    print(f'{best_model_path4 = }')
    
    opts.model = 'UDeepSC_SepCD_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 32
    
    testset, dataloader = get_test_dataloader(opts, infra=False)
    SNRrange = [-6, 12]
    
    metric1 = test_SNR(ta_perform, SNRrange, power_constraint_static, best_model_path1, opts, device, dataloader)
    metric4 = test_SNR(ta_perform, SNRrange, power_constraint, best_model_path4, opts, device, dataloader)

    opts.model = 'UDeepSC_NOMA_new_model'
    metric2 = test_SNR(ta_perform, SNRrange, power_constraint, best_model_path2, opts, device, dataloader)

    opts.model = 'UDeepSC_NOMANoSIC_model'
    metric3 = test_SNR(ta_perform, SNRrange, power_constraint, best_model_path3, opts, device, dataloader)
    
    
    x = [i for i in range(SNRrange[0], SNRrange[1] + 1)]

    models = [metric1, metric4, metric2, metric3]
    # models = [metric1, metric4]

    
    test_set = {
        "Title": "AVE task test model train in homogeneous modal",
        # "Title": "Msa test image & speech",
        "Test Samples": len(testset),
        "power": power_constraint, 
        'udeepsc': str(best_model_path1),
        'perfect SIC': str(best_model_path4),
        'udeepsc NOMA': str(best_model_path2),
        'no SIC': str(best_model_path3),
        'result': models
    }
    save_result_JSON(test_set, result_output)
    
    labels = ["U-DeepSC", "U-DeepSC NOMA (perfect SIC)", "U-DeepSC NOMA (with SIC)", "U-DeepSC NOMA (w/o SIC)"]
    draw_line_chart(x, models, 
                    # y_lim= chart_args['y_lim'], 
                    labels=labels, 
                    title=chart_args['channel_type'], 
                    xlabel="SNR/dB", 
                    ylabel=chart_args['y_axis'], 
                    output=chart_args['output'])

def main_test1(test_bleu=False):
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
    best_model_path = get_best_checkpoint(folder, 'snr12new')
    print(f'{best_model_path = }')
    opts.model = 'UDeepSC_new_model'
    opts.resume = best_model_path
    opts.ta_perform = ta_perform
    
    model = get_model(opts)
    print(f'{opts.resume = }')
    checkpoint_model = load_checkpoint(model, opts)
    load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    
    # SNR for testing, default = 12
    # range: [-6, 12]
    test_snr = torch.FloatTensor([12])

    # using europarl dataset
    opts.textr_euro = True
    dataset_N = "Europarl"
    
    if(test_bleu):
        testset, dataloader = get_test_dataloader(opts, batch_size=32)

        
        test_stats = text_test_BLEU(ta_perform, dataloader, test_snr, model, device)
        
        def draw_BLEU_chart(bleus):
            plt.figure(figsize=(13, 7))  # Set the figure size
            
            x = [i for i in range(len(testset))]
            
            plt.plot(x, bleus, linestyle='-', label="Bleu score")
            
            x_tick = np.arange(x[0], x[-1] + 1, 500)
            
            plt.title(f"BLEU Score for {dataset_N} Dataset", fontsize = 14)
            plt.xticks(x_tick, labels=x_tick)
            plt.xlabel("Index", fontsize=12)
            plt.ylabel("Bleu score", fontsize=12)
            # Modify tick label size
            plt.tick_params(axis='both', which='major', labelsize=14)
            
            plt.grid(True)  # Add grid lines
            
            # Add a legend to distinguish the lines
            plt.legend()
            plt.tight_layout()  # Adjust layout to fit elements properly
            
            plt.savefig(f"bleu_{dataset_N}_new" + '.png') 

        draw_BLEU_chart(test_stats['bleu'])
        draw_threshold_bars(test_stats['bleu'], 0.9, "Distribution of BLEU score", "BLEU Score", "Count", f"bleu_count_{dataset_N}")
        print(f"Average BLEU on the {len(testset)} test samples: {test_stats['average']:.3f}")
    
        return
    
    inputs, _ = get_test_samples(opts, batch_size=30)
    
    received = text_test(ta_perform, inputs, test_snr, model, device)
    
def main_test_shift():
    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:0'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [0.5, 1.0, 1.5]

    result_output = ta_perform + "_result_shift_diff"
    root = './output'
    models_dir = Path(root)
    
    chart_args = {
        'channel_type' : "AWGN channel (SNR = 0)",
        'output': "acc_" + ta_perform + "_result" + "_shift_diff",
        'y_axis': "Accuracy (%)",
        "y_lim" : [55, 83, 5],
        # "y_lim" : [20, 62, 10],
        "vqa_upper": 55.8,
        "msa_upper": 83
    }
    
    root = './output'
    models_dir = Path(root)
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'
    
    # udeepsc
    best_model_path1 = get_best_checkpoint(folder, "udeepscM3Old")
    print(f'{best_model_path1 = }')
    
    # udeepsc Noma
    best_model_path2 = get_best_checkpoint(folderSIC, "CRSIC")
    print(f'{best_model_path2 = }')
    
    # udeepsc Noma no SIC
    best_model_path3 = get_best_checkpoint(folder_noSIC, "powerSumOld")
    print(f'{best_model_path3 = }')

    # udeepsc Noma perfect SIC
    best_model_path4 = get_best_checkpoint(folder_pfSIC, "udeepscM3Old")
    print(f'{best_model_path4 = }')
    
    opts.ta_perform = ta_perform
    opts.batch_size = 10
    testset_size = 4654
    test_shifts = list(range(0, 501, 50)) + list(range(1000, testset_size + 1, 1000))
    snr = 0
    
    opts.model = 'UDeepSC_SepCD_model'
    metric1 = test_shift_data(ta_perform, test_shifts, power_constraint_static, best_model_path1, opts, device, snr)
    metric4 = test_shift_data(ta_perform, test_shifts, power_constraint, best_model_path4, opts, device, snr)

    opts.model = 'UDeepSC_NOMA_new_model'
    metric2 = test_shift_data(ta_perform, test_shifts, power_constraint, best_model_path2, opts, device, snr)
    
    # power_constraint = [3, 3, 3]
    opts.model = 'UDeepSC_NOMANoSIC_model'
    metric3 = test_shift_data(ta_perform, test_shifts, power_constraint, best_model_path3, opts, device, snr)
    
    # print(snr12_bleus)
    
    models = [metric1, metric4, metric2, metric3]
    
    test_set = {
        "Title": "MSA shift test, fix image",
        "SNR": snr,
        "power": power_constraint, 
        'udeepsc': str(best_model_path1),
        'perfect SIC': str(best_model_path4),
        'udeepsc NOMA': str(best_model_path2),
        'no SIC': str(best_model_path3),
        'result': models
    }
    
    save_result_JSON(test_set, result_output)

    labels = ["U-DeepSC", "U-DeepSC with perfect SIC", "U-DeepSC NOMA (with SIC)", "U-DeepSC NOMA (w/o SIC)"]
    # draw_line_chart(test_shifts, models, 
    #                 y_lim= chart_args['y_lim'],
    #                 labels=labels, 
    #                 title=chart_args['channel_type'], 
    #                 xlabel="Shifts", 
    #                 ylabel=chart_args['y_axis'], 
    #                 output=chart_args['output'],
    #                 )

def main_test_draw_from_read():
    ta = 'vqa'
    file_name = "result/" + "vqa_result0205-1806.json"
    chart_args = {
        'channel_type' : "AWGN channel",
        'output': "acc_" + ta,
        'y_axis': "Accuracy (%)",
        # "y_lim" : [55, 82, 5],
        "y_lim" : [20, 62, 10],
        "vqa_upper": 55.8,
        "msa_upper": 81
    }
    
    # Load the list back
    with open(file_name, "r") as file:
        models = json.load(file)
    
    SNRrange = [-6, 12]
    x = [i for i in range(SNRrange[0], SNRrange[1] + 1)]
    upper_bound = [chart_args[f'{ta}_upper']] * len(x)
    models = [upper_bound] + models
    
    labels = ["Upper bound", "U-DeepSC", "U-DeepSC with perfect SIC", "U-DeepSC NOMA (with SIC)", "U-DeepSC NOMA (w/o SIC)"]
    draw_line_chart(x, models, y_lim= chart_args['y_lim'], labels=labels, title=chart_args['channel_type'], xlabel="SNR/dB", ylabel=chart_args['y_axis'], output=chart_args['output'])
    
if __name__ == '__main__':
    # main_test1()
    # main_test1(True)
    main_test_SNR()
    # main_test_SNR_single()
    # main_test_Modal_SNR()
    # main_test_signals()
    # main_test_draw_from_read()
    # main_test_shift()
