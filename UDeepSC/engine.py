import torch
import math
import nltk
import torch.nn as nn
import sys

from utils import *
from tqdm import tqdm
from timm.data import Mixup
from einops import rearrange
from typing import Iterable, Optional
from vqa_utils import VQA_Tool, VQA_Eval
from timm.utils import accuracy, AverageMeter
from nltk.translate.bleu_score import sentence_bleu
from optim_factory import create_optimizer
####################################

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

@torch.no_grad()
def evaluate(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, power_constrain: list[float], print_freq=10):
    net.eval()
    if ta_perform.startswith('imgc'):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform)
                loss = criterion(outputs, targets)
                batch_size = targets.size(0)
                idx, predicted = outputs.max(1)
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
                loss_meter.update(loss.item(), 1)
                if batch_idx % print_freq == 0:
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' %(batch_idx*batch_size, 
                            len(dataloader.dataset), loss_meter.avg, acc_meter.avg*100))   
        test_stat = {'loss': loss_meter.avg,
            'acc': acc_meter.avg}  
        return test_stat
    
    elif ta_perform.startswith('imgr'):
        psnr_meter = AverageMeter()
        loss_meter = AverageMeter()
        psnr_list = []
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform)
                outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
                outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
                loss = criterion(outputs, targets)
                batch_size = targets.shape[0]
                ######  Predictions  ######
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                targets = torch.chunk(imgs, chunks=imgs.size(0), dim=0)
                psnr_vals = calc_psnr(predictions, targets)
                psnr_list.extend(psnr_vals)
                psnr_meter.update(torch.mean(torch.tensor(psnr_vals)).item(), n=batch_size)
                loss_meter.update(loss.item(), 1)
                if batch_idx % print_freq == 0:
                    print('Test %d/%d: [loss: %.4f] [psnr: %.3f dB]' %(batch_idx*batch_size, 
                            len(dataloader.dataset), loss_meter.avg, psnr_meter.avg))   
        test_stat = {'loss': loss_meter.avg,
            'psnr': psnr_meter.avg}  
        return test_stat
    
    elif ta_perform.startswith('textc'):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (texts, targets) in enumerate(dataloader):
                texts, targets = texts.to(device), targets.to(device)
                outputs = net(text=texts, ta_perform=ta_perform)
                loss = criterion(outputs, targets)
                batch_size = targets.size(0)
                idx, predicted = outputs.max(1)
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
                loss_meter.update(loss.item(), 1)
                if batch_idx % print_freq == 0:
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' %(batch_idx*batch_size, 
                            len(dataloader.dataset), loss_meter.avg, acc_meter.avg*100))   
        test_stat = {'loss': loss_meter.avg,
            'acc': acc_meter.avg}  
        return test_stat
    
    elif ta_perform.startswith('textr'):
        bleu_meter = AverageMeter()
        loss_meter = AverageMeter()
        result = []
        with torch.no_grad():
            for batch_idx, (texts, targets) in enumerate(dataloader):
                loss = 0
                texts, targets = texts.to(device), targets.to(device)
                targets = targets[:,1:]
                outputs = net(text=texts, ta_perform=ta_perform)
                batch_size = targets.size(0)
                preds = torch.zeros_like(targets)
                for i in range(outputs.shape[1]):
                    # print(targets.shape)
                    # print(outputs.shape)
                    loss += criterion(outputs[:,i], targets[:,i])
                    preds[:,i] = outputs[:,i].max(-1)[-1] 

                preds = tokens2sentence(preds)
                targets = tokens2sentence(targets)
                for pred, target in zip(preds, targets):
                    # print(f'Pred: {pred}')
                    # print(f'Tar: {target}')
                    result.append((pred, target))
        
                bleu_meter.update(computebleu(preds, targets)/batch_size, n=batch_size)
                loss_meter.update(loss.item(), 1)
                if batch_idx % print_freq == 0:
                    print('Test %d/%d: [loss: %.4f] [bleu: %.3f]' %(batch_idx*batch_size, 
                            len(dataloader.dataset), loss_meter.avg, bleu_meter.avg))   
        test_stat = {'loss': loss_meter.avg,
            'bleu': bleu_meter.avg}  
        return test_stat

@torch.no_grad()
def evaluate_vqa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, power_constraint: list[float], print_freq=500):
    net.eval()
    dataset = dataloader.dataset
    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    i = 0
    for batch_idx, (imgs, texts, targets) in enumerate(dataloader):
        imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
        batch_size = imgs.shape[0]
        i += batch_size
        outputs = net(img=imgs, text=texts, ta_perform=ta_perform, power_constraint=power_constraint)
        pred_np = outputs.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        if pred_argmax.shape[0] != dataset.configs.eval_batch_size:
            pred_argmax = np.pad(
                pred_argmax,(0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
                mode='constant',constant_values=-1)
        ans_ix_list.append(pred_argmax)
        if batch_idx % print_freq == 0:
            print('Test %d/%d:' %(batch_idx*batch_size, 
                        len(dataloader.dataset)))
        
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

    return vqaEval.accuracy

def get_channel_loss(criterion, outputs, targets):
    total_loss = 0.0
    for out, tar in zip(outputs, targets):
        total_loss += criterion(out, tar)
        
    return total_loss
        
def train_channel_batch_uni(ta_perform, model, sel_batch, targets, criterion, power_constraint: list[float]):
    loss = 0
    imgs, texts, speechs = sel_batch
    channel_criterion = criterion['channel_decoder']
    
    if ta_perform.startswith('imgc'):
        targets, Tx_sigs = model.get_semantic_signals(img=imgs,ta_perform=ta_perform, power_constraint=power_constraint)
        Rx_sigs = [model.detecter.img_channel_decoder(Tx_sigs[0])]
        
        loss = get_channel_loss(channel_criterion, Rx_sigs, targets)
    elif ta_perform.startswith('textc'):
        targets, Tx_sigs = model.get_semantic_signals(text=texts, ta_perform=ta_perform,power_constraint=power_constraint)[0]
        Rx_sigs = [model.detecter.text_channel_decoder(Tx_sigs[0])]
        
        loss = get_channel_loss(channel_criterion, Rx_sigs, targets)
    elif ta_perform.startswith('vqa'):
        targets, Tx_sigs = model.get_semantic_signals(img=imgs, text=texts, ta_perform=ta_perform,power_constraint=power_constraint)
        Rx_img_sig = model.detecter.img_channel_decoder(Tx_sigs[0])
        Rx_text_sig = model.detecter.text_channel_decoder(Tx_sigs[1])
        Rx_sigs = [Rx_img_sig, Rx_text_sig]
        
        loss = get_channel_loss(channel_criterion, Rx_sigs, targets)
    elif ta_perform.startswith('msa'):
        targets, Tx_sigs = model.get_semantic_signals(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform,power_constraint=power_constraint)
        Rx_img_sig = model.detecter.img_channel_decoder(Tx_sigs[0])
        Rx_text_sig = model.detecter.text_channel_decoder(Tx_sigs[1])
        Rx_spe_sig = model.detecter.spe_channel_decoder(Tx_sigs[2])
        Rx_sigs = [Rx_img_sig, Rx_text_sig, Rx_spe_sig]
        
        loss = get_channel_loss(channel_criterion, Rx_sigs, targets)
    
    return loss
    


def train_class_batch_uni(ta_perform, model, sel_batch, targets, criterion, power_constraint: list[float]):
    loss = 0
    imgs, texts, speechs = sel_batch
    """
    Bert model input parameters: input_ids, attention_mask, token_type_ids...
    If use pretrained bert on huggingface, there is no "ta_perform" parameter.
    """
    if ta_perform.startswith('imgc'):
        outputs = model(img=imgs,ta_perform=ta_perform, power_constraint=power_constraint)
        loss = criterion[ta_perform](outputs, targets) * 1  
    elif ta_perform.startswith('imgr'):
        outputs = model(img=imgs, ta_perform=ta_perform,power_constraint=power_constraint) 
        targets = rearrange(targets, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        loss = criterion[ta_perform](outputs, targets) * 30
    elif ta_perform.startswith('textc'):
        outputs = model(text=texts, ta_perform=ta_perform,power_constraint=power_constraint)
        loss = criterion[ta_perform](outputs, targets) * 0.6
    elif ta_perform.startswith('textr'):
        outputs = model(text=texts, ta_perform=ta_perform,power_constraint=power_constraint) * 1
        targets = targets[:,1:]
        # print(targets.shape)
        # print(outputs.shape)
        for i in range(outputs.shape[1]):
            loss += criterion[ta_perform](outputs[:,i], targets[:,i])*5
    elif ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform,power_constraint=power_constraint)
        loss = criterion[ta_perform](outputs, targets) * 3
    elif ta_perform.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform,power_constraint=power_constraint)
        loss = criterion[ta_perform](outputs, targets) * 8
    return loss, outputs

def meter(ta_sel):
    acc_meter_dict = {}
    acc_meter_dict['imgc'] = AverageMeter()
    acc_meter_dict['textc'] = AverageMeter()
    acc_meter_dict['vqa'] = AverageMeter()

    loss_meter_dict = {}
    for ta in ta_sel:
        loss_meter_dict[ta] = AverageMeter()
    psnr_meter = AverageMeter()
    return acc_meter_dict, loss_meter_dict, psnr_meter

def train_channel_epoch_uni(model: torch.nn.Module, criterion: dict,
                data_dict: dict, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_sel, power_constraint:list[float], max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=10):
    """
        Training phase 1 for training channel decoder in SIC of UDeepSC_NOMA_model
    """
    
    model.train(True)
    loss_meter =  AverageMeter()                                          

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    num_samples = 5000
    data_iter_step = 0
    num_tasks = len(data_dict)
    data_tuple = [data_loader for data_loader in data_dict.values()]
    # data_tuple[2],data_tuple[3],data_tuple[4]
    train_stat = {}
        
    for data_batch in zip(*data_tuple):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    lr_scale = param_group.get("lr_scale", 1.0)  # Use default value if "lr_scale" is missing
                    param_group["lr"] = lr_schedule_values[it] * lr_scale              
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        imgs, texts, speechs, targets = None, None, None, None
        ta_index = np.random.randint(num_tasks)
        ta = ta_sel[ta_index]
        data = data_batch[ta_index]
        if ta.startswith('img'):
            imgs = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('text'):
            texts = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('vqa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            targets = data[2].to(device, non_blocking=True)
        elif ta.startswith('msa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            speechs = data[2].to(device, non_blocking=True)
            targets = data[3].to(device, non_blocking=True)
        else:
            raise NotImplementedError()
        batch_size = targets.shape[0]
        sel_batch = [imgs, texts, speechs]                                           
        # with torch.cuda.amp.autocast():
        
        chDecoder_loss = train_channel_batch_uni(
        ta, model, sel_batch, targets, criterion, power_constraint)
        chDecoder_loss_val = chDecoder_loss.item()

        # print(loss)
        ######  Error                              
        if not math.isfinite(chDecoder_loss_val):   
                print("Channel loss is {}, stopping training".format(chDecoder_loss_val))
                sys.exit(1)
        ######  Update
        
        if loss_scaler is None:
            chDecoder_loss /= update_freq
            chDecoder_loss.backward()
            model.detecter.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            chDecoder_loss /= update_freq
            grad_norm = loss_scaler(chDecoder_loss, optimizer, clip_grad=max_norm,
                                    parameters=model.detecter.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    
        data_iter_step += 1
        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])
            
        loss_meter.update(chDecoder_loss_val, 1)
        if data_iter_step % print_freq == 0:
            print('Epoch:[%d] [%s] %d/%d: [loss: %.3f][lr: %.3e]' 
                %(epoch, "SIC", batch_size*data_iter_step, 5000,
                    loss_meter.avg, max_lr))
        
    return loss_meter.avg


def train_epoch_uni(model: torch.nn.Module, criterion: dict,
                data_dict: dict, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_sel, power_constraint:list[float], max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=10):
    '''
        Return:
            a dict of average loss for each task in 'ta_sel'
    '''
    model.train(True)                                                         
    acc_meter_dict, loss_meter_dict, psnr_meter = meter(ta_sel)

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    num_samples = 5000
    data_iter_step = 0
    num_tasks = len(data_dict)
    data_tuple = [data_loader for data_loader in data_dict.values()]
    # data_tuple[2],data_tuple[3],data_tuple[4]
    train_stat = {}
        
    for data_batch in zip(*data_tuple):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        imgs, texts, speechs, targets = None, None, None, None
        ta_index = np.random.randint(num_tasks)
        ta = ta_sel[ta_index]
        data = data_batch[ta_index]
        if ta.startswith('img'):
            imgs = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('text'):
            texts = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('vqa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            targets = data[2].to(device, non_blocking=True)
        elif ta.startswith('msa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            speechs = data[2].to(device, non_blocking=True)
            targets = data[3].to(device, non_blocking=True)
        else:
            raise NotImplementedError()
        batch_size = targets.shape[0]
        sel_batch = [imgs, texts, speechs]                                           
        # with torch.cuda.amp.autocast():
        loss, outputs = train_class_batch_uni(
            ta, model, sel_batch, targets, criterion, power_constraint)
        loss_value = loss.item()
        # print(loss)
        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    
        data_iter_step += 1
        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta.endswith('c'):
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('imgr'):
            outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
            tr_imgs = torch.tensor((imgs*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
            re_imgs = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
            mse_cal = nn.MSELoss()
            psnr_meter.update(10 * math.log10(255.0**2/(mse_cal(tr_imgs, re_imgs))), n=1)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('textr'):
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('vqa'):
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('msa'):
            # acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta.startswith('imgc'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, 5000,
                        loss_meter_dict[ta].avg, acc_meter_dict[ta].avg*100, max_lr))
            elif ta.startswith('imgr'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [psnr: %.3f dB] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, 5000,
                        loss_meter_dict[ta].avg, psnr_meter.avg, max_lr)) 
            elif ta.startswith('textc'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, num_samples,
                        loss_meter_dict[ta].avg, acc_meter_dict[ta].avg*100, max_lr)) 
            elif ta.startswith('textr'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, num_samples,
                        loss_meter_dict[ta].avg, max_lr)) 
            elif ta.startswith('vqa'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, 5000,
                        loss_meter_dict[ta].avg, acc_meter_dict[ta].avg*100, max_lr))
            elif ta.startswith('msa'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]' 
                    %(epoch, ta, batch_size*data_iter_step, 5000,
                        loss_meter_dict[ta].avg,  max_lr))
            
    train_stat[ta] = loss_meter_dict[ta].avg  


    return train_stat 


def train_class_batch_vqa(ta_perform, model, imgs, texts, targets, criterion):
    if ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform)
        loss = criterion(outputs, targets)
    return loss, outputs


def train_epoch_vqa(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=500):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (imgs, texts, targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = imgs.size(0)        
                           
        # with torch.cuda.amp.autocast():
        loss, outputs = train_class_batch_vqa(
                ta_perform, model, imgs, texts, targets, criterion)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta_perform.startswith('vqa'):
            acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('vqa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, acc_meter.avg*100, max_lr))
              
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 





@torch.no_grad()
def evaluate_msa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, power_constraint:list[float], print_freq=10):
    net.eval()
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_idx, (imgs,texts,speechs, targets) in enumerate(dataloader):
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
            outputs = net(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, power_constraint=power_constraint)
            loss = criterion(outputs, targets)
            y_pred.append(outputs.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
            loss_meter.update(loss.item(), 1)
    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred = np.concatenate(y_pred, axis=0).squeeze()
    acc = calc_metrics(y_true, y_pred)        
    test_stat = {'loss':loss_meter.avg,
                 'acc': acc}
    return test_stat
    




def train_class_batch_msa(ta_perform, model, imgs, texts, speechs, targets, criterion):
    if ta_perform.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)
        loss = criterion(outputs, targets)
        # pass
    return loss, outputs

def train_epoch_msa(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=5):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (imgs, texts, speechs, targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        speechs = speechs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = imgs.size(0)        
                           
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch_msa(
                ta_perform, model, imgs, texts, speechs, targets, criterion)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta_perform.startswith('msa'):
            # acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('msa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg,  max_lr))
              
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 
