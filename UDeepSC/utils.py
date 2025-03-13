import io
import os
import csv
import math
import time
import json
from typing import Iterable
import thop
import torch
import datetime
import numpy as np
import torch.distributed as dist
import colorama
import wandb

from pathlib import Path
from torch import inf
import torch.nn.functional as F
from timm.utils import get_state_dict
from timm.models import create_model
from collections import OrderedDict
from pytorch_msssim import ms_ssim, ssim
import matplotlib.pyplot as plt
from timm.loss import LabelSmoothingCrossEntropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, f1_score
## Including pakages
def sel_criterion_train(args, ta_sel, device):
    criterion_group = {}
    for ta in ta_sel:
        if ta.startswith('imgc'):
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('textc'):
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('imgr'):
            criterion = torch.nn.MSELoss().to(device)
            print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('textr'):
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            # criterion = torch.nn.CrossEntropyLoss().to(device)
            print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('vqa'):
            criterion = torch.nn.BCELoss(reduction='sum').to(device)
            print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('msa'):
            criterion = torch.nn.MSELoss().to(device)
            print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
        elif ta.startswith('ave'):
            criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
            print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
        else:
            raise NotImplementedError()
        criterion_group[ta] = criterion

    return criterion_group


def sel_criterion_test(args,device):
    if args.ta_perform.startswith('imgc'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('textc'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('imgr'):
        criterion = torch.nn.MSELoss().to(device)
        print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('textr'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('vqa'):
        criterion = torch.nn.BCELoss(reduction='sum').to(device)
        print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('msa'):
        criterion = torch.nn.MSELoss().to(device)
        print("criterion for %s Reconstruction = %s" % (args.ta_perform,str(criterion)))
    elif args.ta_perform.startswith('ave'):
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
        print("criterion for %s classification = %s" % (args.ta_perform,str(criterion)))
    else:
        raise NotImplementedError()

    return criterion

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
 
     
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=> Number of params: {} M'.format(n_parameters / 1e6))
    
    return model

def load_checkpoint(model,args):
    checkpoint = torch.load(args.resume, map_location='cpu')

    print("Load ckpt from the place")
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    # for k in ['head.weight', 'head.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        # print(key)
        if key.startswith('encoder.'):
            new_dict['img_'+key] = checkpoint_model[key]
        # elif key.startswith('text'):
        #     continue
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    return checkpoint_model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
    
def draw_line_chart(x, y_lists, y_lim:list[float]=None, labels=None, title="Line Chart", xlabel="X-axis", ylabel="Y-axis", output="plot"):
    """
    Draws a line chart using two lists for x and y coordinates.

    Inputs:
        x (list): Values for the x-axis.
        y_lists (list of lists): A list of lists where each inner list represents y-values for one line.
        y_lim: min and max y-axis ticks [min, max]
        labels (list): Labels for each line. Default is None, which will use a generic label.
        title (str): Title of the chart. Default is "Line Chart".
        xlabel (str): Label for the x-axis. Default is "X-axis".
        ylabel (str): Label for the y-axis. Default is "Y-axis".
    """
    if not all(len(x) == len(y) for y in y_lists):
        raise ValueError("All y-lists must have the same length as the x-list.")
    
    plt.figure(figsize=(6, 7))  # Set the figure size
    
    # Plot each y-list against the x-list
    for i, y in enumerate(y_lists):
        label = labels[i] if labels and i < len(labels) else f"Line {i+1}"
        plt.plot(x, y, marker='o', linestyle='-', label=label)  # Plot each line
    
    # Set titles and labels
    x_tick = np.arange(x[0], x[-1] + 1, 2)
    
    if(y_lim):
        y_tick = np.arange(y_lim[0], y_lim[1] + 1, y_lim[2])
        plt.yticks(y_tick, labels=y_tick)
    
    plt.title(title, fontsize = 16)
    plt.xticks(x_tick, labels=x_tick)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # Modify tick label size
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.grid(True)  # Add grid lines
    
    # Add a legend to distinguish the lines
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit elements properly
    # plt.show() # on workstation, plt.show() might deadlock due to the workstation
    
    plt.savefig(output + '.png')
    

def draw_threshold_bars(data, threshold, title, xlabel,  ylabel, output):
    plt.figure(figsize=(6, 7))  # Set the figure size
    
    # 計算大於 threshold 和小於 threshold 的數量
    greater_than_threshold = sum(1 for value in data if value > threshold)
    less_than_threshold = len(data) - greater_than_threshold  # 小於 threshold 的數量

    # 繪製柱狀圖
    plt.bar([f'Greater than {threshold}', f'Less than {threshold}'], [greater_than_threshold, less_than_threshold], color=['y', 'y'], width=0.5)

    # 標題和標籤
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 顯示圖表
    plt.savefig(output + '.png')

def train_log(epoch, loss_all:dict):
    log_data = {"epoch": epoch}
    for ta, loss in loss_all.items():
        log_data[f"{ta}/training_loss"] = loss

    wandb.log(log_data)
    
def validation_log(ta_perform, epoch, stats):
    log_data = {"epoch": epoch}
    if ta_perform.startswith('vqa'):
        log_data[f'{ta_perform}/accuracy'] = stats['overall']
        for ansType in stats['perAnswerType']:
            log_data[f'{ta_perform}/{ansType}/accuracy'] = stats['perAnswerType'][ansType]
    else: 
        log_data[f'{ta_perform}/valid_loss'] = stats['loss']
        metric = list(stats.keys())[1]
        log_data[f'{ta_perform}/{metric}'] = stats[metric]
        
    wandb.log(log_data)
    
def toColor(text: str, color: str, other: str='') -> str:
    """
        Make a colored (ANSI) string.
        
        Args:
            text: your stuff. Can be anything, will be str()-ed
            color: the color of your text, must be colorama supported. 
                   e.g. 'yellow', 'cyan'
            other: other attribute that you wanna add to the string
                   e.g. colorama.Style.BRIGHT
        
        Returns:
            An ANSI-colored string.
    """
    return f'{getattr(colorama.Fore, color.upper())}{other}{text}{colorama.Style.RESET_ALL}'

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer) ## update parameters
            self._scaler.update()        ## update scaler state
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def path_exists_make(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir+'/ckpt_'+args.ta_perform)
    path_exists_make(output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor
    
    
def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

# def batch_index_select(x, idx):
#     if len(x.size()) == 3:
#         B, N, C = x.size()
#         N_new = idx.size(1)
#         offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
#         idx = idx + offset
#         out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
#         return out
#     elif len(x.size()) == 2:
#         B, N = x.size()
#         N_new = idx.size(1)
#         offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
#         idx = idx + offset
#         out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
#         return out
#     else:
#         raise NotImplementedError


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)

    if mse == 0:
        return torch.tensor([100.0])

    PIXEL_MAX = 255.0

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def get_imagenet_list(path):
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fns.append(row[0])
    
    return fns

def complex_sig(shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

def pwr_normalize(sig):
    _, num_ele = sig.shape[0], torch.numel(sig[0])
    pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
    sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))
    return sig

def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, c
    img = np.swapaxes(img, 0, 2)  # c, h, w
    return torch.from_numpy(img).float()

def to_chan_last(img):
    img = img.transpose(1, 2)
    img = img.transpose(2, 3)
    return img

def as_img_array(image):
    image = image.clamp(0, 1) * 255.0
    return torch.round(image)

def calc_psnr(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = psnr(original, compare)
        metric.append(val)
    return metric

def calc_msssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        # val = msssim(original, compare)
        val = ms_ssim(original, compare, data_range=255,
                      win_size=3, size_average=True)
        metric.append(val)
    return metric

def calc_ssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = ssim(original, compare, data_range=255,
                   size_average=True)
        metric.append(val)
    return metric

import nltk
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokens2sentence(outputs):
    """
        This methode decode token by token
    """
    sentences = []
    #print(outputs)
    for tokens in outputs:
        sentence = []
        for token in tokens:
            
            word = tokenizer.decode([int(token)])
 
            if word == '[PAD]':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences  
 
def tokens2sentenceV2(outputs):
    """
        This methode decode whole sentence of tokens into human-readable text without the "##" at a time
    """
    sentences = []
    #print(outputs)
    for tokens in outputs:
        tokens = tokens.tolist()
        tokens = tokenizer.decode(tokens)
        sentence = tokens[0].split(" ")
        sentences.append(sentence)
    return sentences 
 
def computebleu(sentences, targets):
  score = 0 
  assert (len(sentences) == len(targets))
  def cut_token(sentence):
    tmp = []
    for token in sentence:
      if token == '[UNK]':
        tmp.append(token)
      else:
        tmp += [word for word in token]
    return tmp 

  for sentence, target in zip(sentences, targets):
    sentence = cut_token(sentence)
   
    target = cut_token(target)

    score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          
  return score

def computBertScore(sentences, targets):
    assert (len(sentences) == len(targets))
    P, R, F1 = score(sentences, targets, lang="en", verbose=True)                     
    return F1

def calc_metrics(y_true, y_pred, mode=None, to_print=True):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    
    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    # pos - neg
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    if to_print:
        # print("mae: ", mae)
        # print("corr: ", corr)
        # print("mult_acc: ", mult_a7)
        print("Classification Report (pos/neg) :")
        # print(classification_report(binary_truth, binary_preds, digits=5))
        print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
        
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        if to_print:
            print("Classification Report (non-neg/neg) :")
            # print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
        
        return accuracy_score(binary_truth, binary_preds)
    
def compute_acc_AVE(labels, x_labels, nb_batch):
    """
        From https://github.com/YapengTian/AVE-ECCV18/blob/master/supervised_main.py: compute_acc
    """
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]): 
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    # target_names = []
    # for i in range(29):
    #     target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)

    
import torch.optim.lr_scheduler as lr_scheduler
def get_scheduler_str(scheduler: lr_scheduler.LRScheduler) -> str:
    """
        Scheduler don't have a __str__ implementation so I asked chatgpt to make one
        I'm guessing it is doing a bad job so there may be exceptions...

        No LambdaLR, MultiplicativeLR because it's not meaningful to print lambda functions
    """
    if isinstance(scheduler, lr_scheduler.StepLR):
        return f'StepLR(optimizer, step_size={scheduler.step_size}, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.MultiStepLR):
        return f'MultiStepLR(optimizer, milestones={scheduler.milestones}, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.ConstantLR):
        return f'ConstantLR(optimizer, factor={scheduler.factor}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.LinearLR):
        return f'LinearLR(optimizer, start_factor={scheduler.start_factor}, end_factor={scheduler.end_factor}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.ExponentialLR):
        return f'ExponentialLR(optimizer, gamma={scheduler.gamma})'
    
    elif isinstance(scheduler, lr_scheduler.PolynomialLR):
        return f'PolynomialLR(optimizer, power={scheduler.power}, total_iters={scheduler.total_iters})'
    
    elif isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
        return f'CosineAnnealingLR(optimizer, T_max={scheduler.T_max}, eta_min={scheduler.eta_min})'
    
    elif isinstance(scheduler, lr_scheduler.ChainedScheduler):
        return f'ChainedScheduler(schedulers={[get_scheduler_str(s) for s in scheduler._schedulers]})'
    
    elif isinstance(scheduler, lr_scheduler.SequentialLR):
        return f'SequentialLR(optimizer, schedulers={[get_scheduler_str(s) for s in scheduler._schedulers]}, milestones={scheduler._milestones})'
    
    elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        return (f'ReduceLROnPlateau(optimizer, mode={scheduler.mode}, factor={scheduler.factor}, '
                f'patience={scheduler.patience}, threshold={scheduler.threshold}, threshold_mode={scheduler.threshold_mode})')
    
    elif isinstance(scheduler, lr_scheduler.CyclicLR):
        return (f'CyclicLR(optimizer, base_lr={scheduler.base_lr}, max_lr={scheduler.max_lr}, '
                f'total_size={scheduler.total_size}, step_ratio={scheduler.step_ratio})')
    
    elif isinstance(scheduler, lr_scheduler.OneCycleLR):
        return (f'OneCycleLR(optimizer, _schedule_phases={scheduler._schedule_phases}, total_steps={scheduler.total_steps}, '
                f'epochs={scheduler.epochs})')
    
    elif isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
        return (f'CosineAnnealingWarmRestarts(optimizer, T_0={scheduler.T_0}, T_mult={scheduler.T_mult}, '
                f'eta_min={scheduler.eta_min})')
    return str(scheduler)

def str_type_indent(obj, iter_limit_items: int = 10, dict_limit_items: int = 1000000, array_limit_items: int = -1,
                    explicit_type: bool=False, indent="    ") -> str:
    """
        ref. str_type
    """
    def str_indent(s, indent: str):
        """
            to indent a whole multiline block
            s: the multiline block
        """
        return '\n'.join([f'{indent}{line}' for line in s.split('\n')])

    # ---

    def str_str(obj: str):
        return f'"{obj}"'
    
    def str_direct(obj):
        "for simple things that doesn't need the type to show, e.g., python's int, float, None"
        return f'{obj}'

    def str_object(obj: object):
        "default case, for anything that is not listed below, including np.float32 or something like that"
        return f'{str(obj.__class__)[8:-2]}({obj})'
    
    def str_iterable_inner(obj: Iterable, limit=iter_limit_items, is_obj_str_list: bool = False):
        """
            used when you wanna iterate something, it deals with the indentation and limit

            f"List[len=4]({str_iterable_inner([1, 2, 3, [4, 5]])})"
            -> List[len=4](
                   1, 
                   2, 
                   3, 
                   List[len=2](4, 5)
               )
            
            Args:
                limit: the maximum number of items to show, the rest would be shown as "... (%d more)"
                is_obj_str_list: whether the object is a list of object strings (objects that is already "dfs()"ed)
                                if so, won't dfs again
                                used when you have special stringify function for the iterated object,
                                e.g., dict
        """
        if is_obj_str_list:
            s_ls = obj
        else:
            # stringify the iterated object
            # the if-else is to prevent dfs()ing the object over the limit count
            # since we won't see them in the result anyway
            s_ls = [dfs(o) if _ < limit else None for _, o in enumerate(obj)]
            
        if len(s_ls) == 0:
            # just don't do intent at all
            s = f""
        else:
            if len(s_ls) > limit:
                # limit the items
                s_ls = s_ls[:limit] + [f'...({len(s_ls) - limit} more)']
            
            if all([type_map.get(o.__class__, None) == str_direct for _, o in zip(range(limit), obj)]):
                # if all items are direct, don't do indentation
                s = ", ".join(s_ls)
            else:
                # do indentation
                inner = ", \n".join(s_ls)
                s = f"\n{str_indent(inner, indent)}\n"
        return s
    
    def str_iterable(obj: Iterable):
        "for default iterables that i don't really know the type of"
        s_ls = [dfs(o) for o in obj]
        s = f"{str(obj.__class__)[8:-2]}[len={len(s_ls)}]({str_iterable_inner(obj)})"
        return s
    
    def str_list(obj: list):
        return f'List[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_tuple(obj: tuple):
        return f'Tuple[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_tensor(obj: torch.Tensor):
        if obj.nelement() <= array_limit_items:
            return f'torch.Tensor[size={list(obj.size())}, dtype={obj.dtype}, dev={obj.device}](\n{str_indent(str(obj), indent)}\n)'
        return f'torch.Tensor[size={list(obj.size())}, dtype={obj.dtype}, dev={obj.device}]()'
    
    def str_np_ndarray(obj: np.ndarray):
        if obj.size <= array_limit_items:
            return f'np.ndarray[shape={list(obj.shape)}, dtype={obj.dtype}](\n{str_indent(str(obj), indent)}\n)'
        return f'np.ndarray[shape={list(obj.shape)}, dtype={obj.dtype}]()'
    
    def str_set(obj: set):
        return f'Set[len={len(obj)}]({str_iterable_inner(obj)})'
    
    def str_dict(obj: set):
        s_ls = [f'{dfs(key)}: {dfs(value)}' for key, value in obj.items()]
        s = f'Dict[len={len(s_ls)}]({str_iterable_inner(s_ls, dict_limit_items, True)})'
        return s
    
    if explicit_type:
        # make simple types also show the type
        str_direct = str_object

    type_map = {
        list: str_list,
        tuple: str_tuple,
        torch.Tensor: str_tensor,
        np.ndarray: str_np_ndarray,
        set: str_set,
        dict: str_dict,
        str: str_str,
        int: str_direct,
        float: str_direct,
        None.__class__: str_direct,
        torch.utils.data.DataLoader: str_direct,
        torch.utils.data.Dataset: str_direct,
        torch.optim.lr_scheduler.LRScheduler: get_scheduler_str,
        Iterable: str_iterable,
    }

    def dfs(obj: object):
        for tp, str_fn in type_map.items():
            if isinstance(obj, tp):
                return str_fn(obj)
        return str_object(obj)

    return dfs(obj)

def str_type(obj, iter_limit_items: int = 10, dict_limit_items: int = 1000000, array_limit_items: int = -1,
             explicit_type: bool=False, indent: int | str | None = None):
    """
        Actually dump everything about... a thing.
        can handle list and tensors and all kinds of stuff.
        useful when you don't know what a thing is and don't wanna just print()
        and see a bunch of tensor values, such as the output of dataloader...

        Args:
            obj: the object to be dumped
            iter_limit_items: the maximum number of items to show in an iterable (other than dict)
                              if the size of the iter is larger than this, will only print up to this amount of items
                              and skip the rest by adding '... (%d more)' at the end
            dict_limit_items: the maximum number of items to show in a dict
            array_limit_items: the maximum number of items to show in an array-like object (i.e., tensor, np.ndarray)
                               if the size of the array is larger than this, no content would be printed
            explicit_type: whether to show the type of simple objects (e.g., int, float, None)
            indent: the indentation of the string. 
                    if int, it will be the number of spaces to indent
                    if str, it will be the string to indent
                    if None, the string will be returned without newlines
    """
    if indent is None:
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, '').replace('\n', '')
    elif isinstance(indent, int):
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, ' ' * indent)
    else:
        return str_type_indent(obj, iter_limit_items, dict_limit_items, array_limit_items, explicit_type, str(indent))

    
class DiffPruningLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, dynamic=True, ratio_weight=2.0, main_weight=1.):
        super().__init__()
        self.base_criterion = base_criterion
        self.main_weight = 1.
        self.surp_weight = 0.022
        self.rho_weight = 0.01    
        self.vq_weight = 2.0    
        self.print_mode = True
        
        self.count = 0
        self.main_loss_record = 0.
        self.surp_loss_record = 0.
        self.vq_loss_record = 0.
        self.keep_ratio_record = 0.
        
        self.dynamic = dynamic
        if self.dynamic:
            print('using dynamic loss')

    def forward(self, outputs, labels):
        pred, mask_m, rho, vq_loss = outputs
        surp_loss = 0.0
        score = mask_m
        keep_ratio = score.mean(1)
   
        surp_loss = surp_loss + ((keep_ratio - rho) ** 2).mean()    ### The supervised loss. 
        main_loss = self.base_criterion(pred, labels)              ### Reconstruction loss.

        loss = self.main_weight * main_loss + \
               self.surp_weight * surp_loss + \
               self.rho_weight * rho + self.vq_weight * vq_loss
        # loss = self.clf_weight * cls_loss + vq_loss
        if self.print_mode:
            self.main_loss_record += main_loss.item()
            self.surp_loss_record += surp_loss.item()
            self.vq_loss_record += vq_loss.item()
            self.keep_ratio_record += keep_ratio.mean().item()
            self.count += 1
            if self.count == 100:
                print('loss info: main_loss=%.4f, surp_loss=%.4f, vq_loss=%.4f, keep ratio=%.4f' 
                        % (self.main_loss_record / self.count, 
                           self.surp_loss_record / self.count, 
                           self.vq_loss_record / self.count,
                           self.keep_ratio_record / self.count))
                self.main_loss_record = 0
                self.surp_loss_record = 0
                self.vq_loss_record = 0
                self.keep_ratio_record = 0
                self.count = 0
        return loss