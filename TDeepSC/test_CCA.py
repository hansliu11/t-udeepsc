import numpy as np
from sklearn.cross_decomposition import CCA
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import model
from msa_utils import Config_MSA, MSA
from datasets import collate_fn
from utils import *
from test import get_best_checkpoint
from base_args import get_args

def compute_cca(X, Y, n_components=10):
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    # Return canonical correlation coefficients (Pearson correlation of each component)
    corr_matrix = np.corrcoef(X_c.T, Y_c.T)[:n_components, n_components:]
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    return corrs

def plot_correlation_trend(results, shift_range, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 7))  # Set the figure size
    
    for key, values in results.items():
        plt.plot(range(len(shift_range)), values, marker='*', label=key)
    
    plt.xticks(range(len(shift_range)), labels=[str(a) for a in shift_range], rotation=45)

    y_tick = np.arange(0.5, 1.05, 0.1)
    plt.yticks(y_tick)
    
    plt.xlabel("Offset")
    plt.ylabel("Canonical Correlation")
    plt.title("Correlation across data shift offset")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / 'test_shift_cca.png')

def collect_correlations_rawdata(config, dataset_cls, shift_offsets, batch_size=64, max_samples=3000, device="cpu"):
    from transformers import BertTokenizer, BertModel
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()
    
    results = {
        # 'shift_offsets': shift_offsets,
        'Text-Image': [],
        'Text-Speech': [],
        'Image-Speech': []
    }

    progress = tqdm(shift_offsets, desc="Running CCA over shifts", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for shift in progress:
            dataset = dataset_cls(config, train=False, shift_offset=shift)
            sampler_test = torch.utils.data.SequentialSampler(dataset)
            dataloader= torch.utils.data.DataLoader(
                dataset, sampler=sampler_test, batch_size=int(1.0 * batch_size), shuffle=False, collate_fn=collate_fn)

            text_feats, image_feats, speech_feats = [], [], []
            test_progress = tqdm(dataloader, leave=False, total=len(dataloader), dynamic_ncols=True)
            for (imgs, texts, speechs, _) in test_progress:
                # (text, image, speech, _) = batch
                imgs, texts, speechs = imgs.to(device), texts.to(device), speechs.to(device)

                # === Temporal mean pooling ===
                # print(f"{str_type(imgs_encoded)= }")
                image_pooled = imgs.mean(dim=1)      # [B, D_img]
                speech_pooled = speechs.mean(dim=1)    # [B, D_speech]

                # === BERT encoding + mean pooling ===
                bert_output = bert_model(input_ids=texts)
                text_embeddings = bert_output.last_hidden_state     # [B, L, D]
                text_pooled = text_embeddings.mean(dim=1)           # [B, D]

                # === Append as NumPy arrays ===
                text_feats.append(text_pooled.cpu().numpy())
                image_feats.append(image_pooled.cpu().numpy())
                speech_feats.append(speech_pooled.cpu().numpy())

                if sum(x.shape[0] for x in text_feats) >= max_samples:
                    break

            # Stack all features: shape = [N, D]
            T = np.vstack(text_feats)[:max_samples]
            I = np.vstack(image_feats)[:max_samples]
            S = np.vstack(speech_feats)[:max_samples]

            # Optionally normalize each modality
            T = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-6)
            I = (I - I.mean(axis=0)) / (I.std(axis=0) + 1e-6)
            S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-6)

            # Compute and store mean canonical correlation
            results['Text-Image'].append(np.mean(compute_cca(T, I)))
            results['Text-Speech'].append(np.mean(compute_cca(T, S)))
            results['Image-Speech'].append(np.mean(compute_cca(I, S)))

    return results

def collect_correlations(config, dataset_cls, shift_offsets, model_path, args, batch_size=64, max_samples=3000, device="cpu"):
    args.resume = model_path
    
    model = get_model(args)
    print(f'{args.resume = }')
    checkpoint_model = load_checkpoint(model, args)
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    
    results = {
        # 'shift_offsets': shift_offsets,
        'Text-Image': [],
        'Text-Speech': [],
        'Image-Speech': []
    }

    model.eval()
    progress = tqdm(shift_offsets, desc="Running CCA over shifts", leave=False, dynamic_ncols=True)

    task = args.ta_perform
    with torch.no_grad():
        for shift in progress:
            dataset = dataset_cls(config, train=False, shift_offset=shift)
            sampler_test = torch.utils.data.SequentialSampler(dataset)
            dataloader= torch.utils.data.DataLoader(
                dataset, sampler=sampler_test, batch_size=int(1.0 * batch_size), shuffle=False, collate_fn=collate_fn)

            text_feats, image_feats, speech_feats = [], [], []
            test_progress = tqdm(dataloader, leave=False, total=len(dataloader), dynamic_ncols=True)
            for (imgs, texts, speechs, _) in test_progress:
                # (text, image, speech, _) = batch
                imgs, texts, speechs = imgs.to(device), texts.to(device), speechs.to(device)

                # === Temporal mean pooling ===
                imgs_encoded = model.img_encoder(imgs, task)
                # print(f"{str_type(imgs_encoded)= }")
                image_pooled = imgs_encoded.mean(dim=1)      # [B, D_img]
                spe_encoded = model.spe_encoder(speechs, task)
                speech_pooled = spe_encoded.mean(dim=1)    # [B, D_speech]

                text_encoded = model.text_encoder(input_ids=texts, return_dict=False)[0]
                text_pooled = text_encoded.mean(dim=1)           # [B, D]

                # === Append as NumPy arrays ===
                text_feats.append(text_pooled.cpu().numpy())
                image_feats.append(image_pooled.cpu().numpy())
                speech_feats.append(speech_pooled.cpu().numpy())

                if sum(x.shape[0] for x in text_feats) >= max_samples:
                    break

            # Stack all features: shape = [N, D]
            T = np.vstack(text_feats)[:max_samples]
            I = np.vstack(image_feats)[:max_samples]
            S = np.vstack(speech_feats)[:max_samples]

            # Optionally normalize each modality
            T = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-6)
            I = (I - I.mean(axis=0)) / (I.std(axis=0) + 1e-6)
            S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-6)

            # Compute and store mean canonical correlation
            results['Text-Image'].append(np.mean(compute_cca(T, I)))
            results['Text-Speech'].append(np.mean(compute_cca(T, S)))
            results['Image-Speech'].append(np.mean(compute_cca(I, S)))

    return results

def main_test_msa_CCA():
    testset_size = 4654
    shift_range = list(range(0, 501, 50)) + list(range(1000, testset_size + 1, 1000))
    
    gpus = [0]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{','.join(str(gpu_id) for gpu_id in gpus)}")
    else:
        device = torch.device("cpu")
    
    config_msa = Config_MSA()
    results = collect_correlations_rawdata(config_msa, MSA, shift_range, batch_size=64, device=device)
    
    plot_correlation_trend(results, shift_range, Path("/home/ldap/hansliu/t-udeepsc/TDeepSC/tmp/20250413_cca_test"))

def main_test_msa_CCA_20250413():
    opts = get_args()
    ta_perform = 'msa'

    testset_size = 4654
    # shift_range = list(range(0, 501, 50)) + list(range(1000, testset_size + 1, 1000))
    shift_range = list(range(0, 51, 5))
    
    gpus = [0]
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{','.join(str(gpu_id) for gpu_id in gpus)}")
    else:
        device = torch.device("cpu")
    
    root = './output'
    models_dir = Path(root)
    folder = models_dir / f'ckpt_{ta_perform}'
    
    # udeepsc
    best_model_path = get_best_checkpoint(folder, "checkpoint")
    print(f'{best_model_path = }')
    
    opts.model = f'TDeepSC_{ta_perform}_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 64
    
    config_msa = Config_MSA()
    results = collect_correlations(config_msa, MSA, shift_range, best_model_path, opts, batch_size=64, device=device)
    
    plot_correlation_trend(results, shift_range, Path("/home/ldap/hansliu/t-udeepsc/TDeepSC/tmp/20250416_cca_on_encoder_test"))

if __name__ == "__main__":
    # main_test_msa_CCA()
    main_test_msa_CCA_20250413()