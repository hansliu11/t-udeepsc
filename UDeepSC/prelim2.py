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

import model  
from utils import *
from base_args import get_args
from datasets import build_dataset_test, collate_fn, collate_fn_Shuff
from test import test_SNR

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

def get_regression_expression(reg_model):
    # Extract regression coefficients and print the expression
    lin_reg = reg_model.named_steps['linearregression']
    poly = reg_model.named_steps['polynomialfeatures']
    coef = lin_reg.coef_
    intercept = lin_reg.intercept_
    feature_names = poly.get_feature_names_out(['symbols', 'snr'])

    name2var = {'symbols': 'N', 'snr': '\u03B3'} # symbols: N, snr: γ

    print("Polynomial regression model:")
    print(f"{intercept:.4f}", end=" ")
    for i, (name, c) in enumerate(zip(feature_names[1:], coef[1:]), 1):  # skip bias term
        print(f"+ ({c:.4f})*({name})".replace('symbols', 'N').replace('snr', '\u03B3'), end=" ")
    print()

def plot_3D(output_path, accuracy):
    from mpl_toolkits.mplot3d import Axes3D
    num_symbols = np.arange(1, 25)         # number of symbols: 1 to 24
    snrs = np.arange(-6, 13)               # SNRs: -6 to 12
    snrs_grid, symbols_grid = np.meshgrid(snrs, num_symbols)  # Create coordinate grid

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(snrs_grid, symbols_grid, accuracy, cmap='viridis')

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("N (symbols)")
    ax.set_zlabel("Test Accuracy (%)")
    # ax.set_title("Semantic Communication Accuracy vs Dimension & SNR")
    # Set SNR ticks with step size 5
    ax.set_xticks(np.arange(-6, 13, 2)) 
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

def curve_fitting(output_path, accuracy):
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    # Define axis
    num_symbols = np.arange(1, 25)         # number of symbols: 1 to 24
    snrs = np.arange(-6, 13)               # SNRs: -6 to 12

    # Prepare data for regression
    symbols_grid, snrs_grid = np.meshgrid(num_symbols, snrs, indexing='ij')
    X = np.column_stack([symbols_grid.ravel(), snrs_grid.ravel()])
    y = accuracy.ravel()

    # Fit a polynomial regression model
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)

    # Predict on grid for visualization
    snrs_pred, symbols_pred = np.meshgrid(snrs, num_symbols)
    X_pred = np.column_stack([symbols_pred.ravel(), snrs_pred.ravel()])
    y_pred = poly_model.predict(X_pred).reshape(symbols_pred.shape)

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(snrs_pred, symbols_pred, y_pred, cmap='viridis')
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("N (symbols)")
    ax.set_zlabel("Predicted Accuracy")
    ax.set_title("Polynomial Regression")

    ax.set_xticks(np.arange(-6, 13, 2)) 
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    return poly_model
    
def main_test_different_symbols_SNR():
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    now = now.strftime('%m%d-%H%M')

    opts = get_args()
    ta_perform = 'msa'
    device = 'cuda:1'
    device = torch.device(device)
    power_constraint_static = [1.0, 1.0, 1.0]
    power_constraint = [1.0, 0.5, 1.5]
    # power_constraint = [0.5, 1.5]
    # power_constraint = [0.5]
    print(f"Power Constraint: {power_constraint}")
    
    result_output = ta_perform + "_result" + "_bandwidth2SNR" + now

    opts.model = 'UDeepSC_NOMANoSIC_model'
    opts.ta_perform = ta_perform
    opts.batch_size = 32
    
    testset, dataloader = get_test_dataloader(opts, infra=False, shuffle=False)
    SNRrange = [-6, 12]
    num_symbols = np.arange(1, 25) 
    res = []
    models = {}

    for n in num_symbols:
        opts.num_symbols = n
        root = f'./output/bandwidth/sb{n}'
        models_dir = Path(root)
        folder_noSIC = models_dir / f'noSIC_{ta_perform}'
        best_model_path = get_best_checkpoint(folder_noSIC, "I384_T512_S128")
        print(f'{best_model_path = }')

        metric1 = test_SNR(ta_perform, SNRrange, power_constraint_static, best_model_path, opts, device, dataloader)

        res.append(metric1)
        models[f'symbols_{n}'] = metric1

    
    test_setting = {
        "Title": "Msa test symbols to SNR",
        "Test Samples": len(testset),
        "power": power_constraint, 
        "noSIC power": power_constraint_static,
        "Model": str(best_model_path),
        "Result": models,
        "Array": res,
    }
    
    save_result_JSON(test_setting, result_output)
    print(f"{np.array(res).shape=}")

def main_plot_3D():
    """
        Plot 3D graph for test accuracy vs symbols and SNR
    """
    json_path = "/home/ldap/hansliu/t-udeepsc/UDeepSC/result/msa_result_bandwidth2SNR0721-0835.json"
    jsonFile = open(json_path,'r')
    result = json.load(jsonFile)
    accuracy = np.array(result['Array'])
    print(f"{accuracy.shape=}")

    ta_perform = 'msa'
    result_path = Path(f'./tmp/20250804_Symbols_SNR/{ta_perform}')
    result_path.mkdir(parents=True, exist_ok=True)

    plot_3D(result_path / '3D_plot.png', accuracy)

def main_curve_fitting():
    json_path = "/home/ldap/hansliu/t-udeepsc/UDeepSC/result/msa_result_bandwidth2SNR0721-0835.json"
    jsonFile = open(json_path,'r')
    result = json.load(jsonFile)
    accuracy = np.array(result['Array'])
    print(f"{accuracy.shape=}")

    ta_perform = 'msa'
    result_path = Path(f'./tmp/20250804_Symbols_SNR/{ta_perform}')
    result_path.mkdir(parents=True, exist_ok=True)

    ploy_model = curve_fitting(result_path / 'curve_fitting_degree2.png', accuracy)
    get_regression_expression(ploy_model)

if __name__ == '__main__':
    # main_test_different_symbols_SNR()
    # main_plot_3D()
    main_curve_fitting()