import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from typing import Literal

from transformers import BertTokenizer
from data import CIFAR_CR,SST_CR
from timm.data import create_transform
from vqa_utils import VQA2, Config_VQA
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from msa_utils import PAD, Config_MSA, MSA
from AVE_utils import AVEDataset, Config_AVE
# from pytorch_transformers import BertTokenizer
from torch.utils.data.sampler import RandomSampler
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, number_samp=50000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_samp = number_samp
        self.largest_dataset_size = number_samp

    def __len__(self):
        return self.number_samp

    def __iter__(self):
        sampler = RandomSampler(self.dataset)
        sampler_iterator = sampler.__iter__()
        step = self.batch_size
        samples_to_grab = self.batch_size
        epoch_samples = self.number_samp
        final_samples_list = []  
        ### this is a list of indexes from the combined dataset
        for es in range(0, epoch_samples, step):
            cur_batch_sampler = sampler_iterator
            cur_samples = []
            for eg in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
                except StopIteration:
                    ### got to the end of iterator - restart the iterator and continue to get samples
                    sampler_iterator = sampler.__iter__()
                    cur_batch_sampler = sampler_iterator
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)
        return iter(final_samples_list)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_dataset(is_train, args, split: Literal['val', 'test'] = 'val', infra: bool = False, shifts:int = 0):
    if args.ta_perform.startswith('img'):
        transform = build_img_transform(is_train, args)
        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("------------------------------------------------------")

    if  args.ta_perform.startswith('imgc'):
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                        download=True, if_class=True)
    elif  args.ta_perform.startswith('imgr'):
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                        download=True, if_class=False)
    elif args.ta_perform.startswith('textc'):
        dataset = SST_CR(root=False, train=is_train, binary=True, if_class=True)

    elif args.ta_perform.startswith('textr'):
        dataset = SST_CR(root=True, train=is_train, binary=True, if_class=False)

    elif args.ta_perform.startswith('vqa'):
        config_vqa = Config_VQA()
        config_vqa.proc(args)
        dataset = VQA2(config_vqa, train=is_train)
        
    elif args.ta_perform.startswith('msa'):
        config_msa = Config_MSA()
        dataset = MSA(config_msa, train=is_train, shift_offset=shifts)
    
    elif args.ta_perform.startswith('ave'):
        split = 'train' if is_train else split
        config_ave = Config_AVE()
        dataset = AVEDataset(config_ave, split=split, add_infra=infra)
    
    else:
        raise NotImplementedError()

    return dataset


def build_img_transform(is_train, args):
    resize_im = args.input_size > 32
    mean = (0.,0.,0.)
    std =  (1.,1.,1.)

    t = []
    if is_train:
        if resize_im:
            crop_pct = 1
            size = int(args.input_size / crop_pct)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)  
    targets = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    texts = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    images = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    speechs = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    # print(texts.permute(1,0))
    SENT_LEN = texts.size(0)
    # Create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=SENT_LEN+2, add_special_tokens=True, padding='max_length',truncation=True)
        bert_details.append(encoded_bert_sent)

    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    texts = bert_sentences

    return images.permute(1,0,2), texts, speechs.permute(1,0,2), targets