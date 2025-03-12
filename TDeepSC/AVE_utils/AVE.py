"""AVE dataset"""
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class AVEDataset(Dataset):
    """
        From https://github.com/YapengTian/AVE-ECCV18/blob/master/dataloader.py: AVEDataset
        Translate to pytorch dataset by me
    """
    def __init__(self, config, split):

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        self.data_path = str(config.data_dir)
        self.video_dir = self.data_path + '/visual_feature.h5'
        self.audio_dir = self.data_path + '/audio_feature.h5'
        self.label_dir = self.data_path + '/labels.h5'
        self.order_dir = self.data_path + f'/{split}_order.h5'

        # Load order
        with h5py.File(self.order_dir, 'r') as hf:
            self.order = hf['order'][:]
        
        # Load features and labels
        with h5py.File(self.audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(self.label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(self.video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]

    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, index):
        vid_idx = self.order[index]
        
        video = torch.tensor(self.video_features[vid_idx], dtype=torch.float32) # (time_steps, 7, 7, 512)
        audio = torch.tensor(self.audio_features[vid_idx], dtype=torch.float32) # (time_steps, 128)
        label = torch.tensor(self.labels[vid_idx], dtype=torch.float32) # (time_steps, num_class)
        
        return video, audio, label

    def get_batch(self, idx):
        """ Deprecated """
        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[self.lis[id], :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[self.lis[id], :, :]
            self.label_batch[i, :, :] = self.labels[self.lis[id], :, :]

        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()
    
class Config_AVE(object):
    def __init__(self):
        project_dir = Path(__file__).resolve().parent.parent # .../UDeepSC

        data_dir = project_dir.joinpath('data/avedata')
        self.data_dir = data_dir
        # video_dir = data_dir.joinpath('audio_feature.h5')
