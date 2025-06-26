import torch
from pathlib import Path
from loguru import logger
import re
from transformers import BertTokenizer
from tqdm import tqdm
import json
import datetime
from timeit import default_timer
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from timm.utils import AverageMeter

import model  
from utils import *
from channel import signal_power
from base_args import get_args
from datasets import build_dataset_test, collate_fn, collate_fn_Shuff
from vqa_utils import VQA_Tool, VQA_Eval

def save_result_JSON(results: list, output:str):
    output = "result/" + output + ".json"
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

def compute_estimated_snr(original, reconstructed):
    """
    Args:
        original, reconstructed: torch.Tensor, shape = (batch, 1, feature_dim)
    
    Return:
        SNR (dB) for each batch and average SNR dB over batches
    """
    orig = original.detach().squeeze(1)      # (batch, dim)
    recon = reconstructed.detach().squeeze(1)

    signal_power = torch.mean(orig.abs() ** 2, dim=-1)  # (batch,)
    noise_power = torch.mean((orig - recon) ** 2, dim=-1)
    snr = signal_power / (noise_power + 1e-10)   # 防止除0
    snr_db = 10 * torch.log10(snr)
    avg_snr_db = snr_db.mean().item()
    return snr_db.numpy(), avg_snr_db

def estimated_snr_comparison(user_lists, model_names, output="snr_comparison.png"):
        avg_snrs = [[] for _ in model_names]
        labels = []
        
        for modalities in zip(*user_lists):
            for i, (orig, dec, modal) in enumerate(modalities):
                batch_SNRdb, avg_SNRdb = compute_estimated_snr(orig, dec)
                avg_snrs[i].append(avg_SNRdb)
                if i == 0:
                    labels.append(modal)
        
        x = range(len(labels))
        width = 0.2  # Bar width

        plt.figure(figsize=(10, 6))
        for i, (snrs, model_name) in enumerate(zip(avg_snrs, model_names)):
            bars = plt.bar([j + (i - len(model_names) / 2) * width for j in x], snrs, width, label=model_name)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='top', fontsize=10)

        plt.ylabel('Average SNR (dB)')
        plt.title('Comparison of Average SNR for Each User')
        plt.xticks(x, labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output, dpi=200)

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
        Tx_sigs, Rx_sigs = model.get_features(img=imgs, ta_perform=ta, test_snr=test_snr)
    elif ta.startswith('text'):
        Tx_sigs, Rx_sigs = model.get_features(text=texts, ta_perform=ta, test_snr=test_snr)
    elif ta.startswith('vqa'):
        Tx_sigs, Rx_sigs = model.get_features(img=imgs, text=texts, ta_perform=ta, power_constraint=power_constraint, test_snr=test_snr)
    elif ta.startswith('msa'):
        Tx_sigs, Rx_sigs = model.get_features(img=imgs, text=texts, speech=speechs, ta_perform=ta, power_constraint=power_constraint, test_snr=test_snr)
        # Tx_sigs, Rx_sigs = model.get_features(text=texts, speech=speechs, ta_perform=ta, power_constraint=power_constraint, test_snr=test_snr)
    else:
        raise NotImplementedError()
    
    batch_size = Tx_sigs[0].shape[0]

    # for i in range(batch_size):
    #     for tx, rx in zip(Tx_sigs, Rx_sigs):
    #         print(f"Before transmitting {i}: ")
    #         print(str_type(tx[i]))
    #         print(tx[i])
            
    #         print(f"After transmitting {i}: ")
    #         print(str_type(rx[i]))
    #         print(rx[i])

    #         ## pause
    #         input("press any key..")
        
    return Tx_sigs, Rx_sigs

def main_test_features():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    now = now.strftime('%m%d-%H%M')

    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [1, 0.5, 1.5]
    # power_constraint = [1.6, 0.9, 0.4]
    # power_constraint = [0.5, 1.5]
    result_output = ta_perform + "_features_result_" + now
    
    chart_args = {
        'channel_type' : "AWGN channel",
        'output': "acc_" + ta_perform + "_features_result_",
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
    opts.batch_size = 16
    
    testset, dataloader = get_test_dataloader(opts)
    sel_batch, _ = get_test_samples(opts, opts.batch_size)
    SNRrange = [7, 12]
    

    best_model_path = get_best_checkpoint(folder_pfSIC, f"udeepscM3_1")
    # best_model_path = get_best_checkpoint(folderSIC, f"CRSIC_1")
    
    print(f'{best_model_path = }')
    opts.model = 'UDeepSC_perfectSIC_model'
    # opts.model = 'UDeepSC_NOMA_new_model'

    Tx_sigs, Rx_sigs = test_features(ta_perform, torch.tensor([12]), power_constraint, best_model_path, opts, device, sel_batch)

    def plot_features_line(user_list, model_name, output="plot"):
        for idx, (orig, dec, modal) in enumerate(user_list):
            plt.figure(figsize=(12, 4))
            # batch_idx = 0 
            orig_vec = orig.clone().detach().cpu()
            dec_vec = dec.clone().detach().cpu()
            plt.plot(orig_vec, label="Original", alpha=0.7)
            plt.plot(dec_vec, label="Decoded", alpha=0.7)
            plt.xlabel("Batch Sample Index")
            plt.ylabel("MAE")
            plt.title(modal + f' ({model_name}) ')
            plt.tight_layout()
            plt.savefig(output + modal + '.png')

    def plot_features_heatmap(user_list, model_name, output="plot"):
        for idx, (orig, dec, modal) in enumerate(user_list):
            plt.figure(figsize=(16, 6))
            # batch_idx = 0 
            orig_vec = orig.clone().detach().squeeze(1).cpu()
            dec_vec = dec.clone().detach().squeeze(1).cpu()
            mae = (orig_vec - dec_vec).abs().numpy()  # (batch, dim)
            plt.imshow(mae, aspect='auto', cmap='viridis')
            plt.colorbar(label='MAE')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Batch Index')
            plt.title(modal + f' ({model_name}) ')
            plt.tight_layout()
            plt.savefig(output / (modal + '_heatbar.png'))       

    modality = ['Image', 'Text', 'Speech']
    user_list = []
    for i, f in enumerate(Tx_sigs):
        user_list.append((f, Rx_sigs[i], modality[i]))

    result_path = Path(f'./tmp/20250608_perfectSIC_features/{ta_perform}')
    result_path.mkdir(parents=True, exist_ok=True)

    plot_features_heatmap(user_list, model_name="perfectSIC", output= result_path)


def main_test_features_estimated_snr():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    now = now.strftime('%m%d-%H%M')

    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [1, 0.5, 1.5]
    result_output = ta_perform + "_features_result_" + now
    
    root = './output'
    models_dir = Path(root)
    
    folder = models_dir / f'udeepsc_{ta_perform}'
    folderSIC = models_dir / f'NOMA_{ta_perform}'
    folder_noSIC = models_dir / f'noSIC_{ta_perform}'
    folder_pfSIC = models_dir / f'perfectSIC_{ta_perform}'

    opts.ta_perform = ta_perform
    opts.batch_size = 16
    
    testset, dataloader = get_test_dataloader(opts)
    sel_batch, _ = get_test_samples(opts, opts.batch_size)
    test_SNR = 12

    # Load features for the first model (perfect SIC)
    best_model_path_pfSIC = get_best_checkpoint(folder_pfSIC, f"udeepscM3_1")
    print(f'{best_model_path_pfSIC = }')
    opts.model = 'UDeepSC_perfectSIC_model'
    Tx_sigs_pfSIC, Rx_sigs_pfSIC = test_features(ta_perform, torch.tensor([test_SNR]), power_constraint, best_model_path_pfSIC, opts, device, sel_batch)

    # Load features for the second model (NOMA)
    best_model_path_SIC = get_best_checkpoint(folderSIC, f"CRSIC_1")
    print(f'{best_model_path_SIC = }')
    opts.model = 'UDeepSC_NOMA_new_model'
    Tx_sigs_SIC, Rx_sigs_SIC = test_features(ta_perform, torch.tensor([test_SNR]), power_constraint, best_model_path_SIC, opts, device, sel_batch)

    best_model_path = get_best_checkpoint(folder, f"udeepscM3_1")
    print(f'{best_model_path = }')
    opts.model = 'UDeepSC_SepCD_model'
    Tx_sigs, Rx_sigs = test_features(ta_perform, torch.tensor([test_SNR]), power_constraint, best_model_path, opts, device, sel_batch)

    modality = ['Image', 'Text', 'Speech']
    user_list_pfSIC = [(f, Rx_sigs_pfSIC[i], modality[i]) for i, f in enumerate(Tx_sigs_pfSIC)]
    user_list_SIC = [(f, Rx_sigs_SIC[i], modality[i]) for i, f in enumerate(Tx_sigs_SIC)]
    # user_list_udeepsc = [(f, Rx_sigs[i], modality[i]) for i, f in enumerate(Tx_sigs)]

    result_path = Path(f'./tmp/20250615_features/{ta_perform}')
    result_path.mkdir(parents=True, exist_ok=True)

    estimated_snr_comparison(
        [user_list_pfSIC, user_list_SIC],
        model_names=["perfectSIC", "NOMA SIC"],
        output=result_path / f'snr_{test_SNR}_comparison.png'
    )

def main_test_SIC_TextSpeech_features_estimated_snr():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    now = now.strftime('%m%d-%H%M')

    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0]
    power_constraint = [0.5, 1.5]
    result_output = ta_perform + "_features_result_" + now
    
    root = './output'
    models_dir = Path(root)
    
    folderSIC = models_dir / f'NOMA_{ta_perform}'

    opts.ta_perform = ta_perform
    opts.batch_size = 16
    
    testset, dataloader = get_test_dataloader(opts)
    sel_batch, _ = get_test_samples(opts, opts.batch_size)
    test_SNR = 4

    # Load features for the second model (NOMA)
    best_model_path_SIC = get_best_checkpoint(folderSIC, f"CRSIC_1")
    print(f'{best_model_path_SIC = }')
    opts.model = 'UDeepSC_NOMA_new_model'
    Tx_sigs_SIC, Rx_sigs_SIC = test_features(ta_perform, torch.tensor([test_SNR]), power_constraint, best_model_path_SIC, opts, device, sel_batch)

    modality = ['Text', 'Speech']
    user_list_SIC = [(f, Rx_sigs_SIC[i], modality[i]) for i, f in enumerate(Tx_sigs_SIC)]

    result_path = Path(f'./tmp/20250615_features/{ta_perform}')
    result_path.mkdir(parents=True, exist_ok=True)

    estimated_snr_comparison(
        [user_list_SIC],
        model_names=["NOMA SIC"],
        output=result_path / f'snr_{test_SNR}_comparison_SIC_TextSpe.png'
    )
    

if __name__ == '__main__':
    # main_test_features()
    main_test_features_estimated_snr()
    # main_test_SIC_TextSpeech_features_estimated_snr()