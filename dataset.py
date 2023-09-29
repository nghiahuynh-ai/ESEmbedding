import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ESDataset(Dataset):
    
    def __init__(self, config):

        EMO = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3, 'surprise': 4}
        self.samples = []
        
        for emo in config['dirs'].keys():
            
            emo_dir = config['dirs'][emo]
            if not os.path.isdir(emo_dir):
                raise FileNotFoundError(f'Cannot find the given directory: {emo_dir}')
            
            files = os.listdir(emo_dir)
            for f in files:
                fpath = os.path.join(emo_dir, f)
                if not os.path.isfile(fpath):
                    continue
                self.samples.append((fpath, EMO[emo]))
        
        self.loader = DataLoader(
            self, 
            batch_size=config.batch_size, 
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=Collate(config.sr),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        anchor = self.samples[idx]
        sample_idx = np.random.randint(0, len(self.samples))
        sample = self.samples[sample_idx]
        # return: path1, emo1, path2, emo2
        return anchor[0], anchor[1], sample[0], sample[1]

    
class Collate:
    
    def __init__(self, sr):
        self.sr = sr
        
    def __call__(self, batch):
        
        l_anchor_max, l_sample_max = 0, 0
        anchors, anchors_emo, samples, samples_emo = [], [], [], []
        
        for anchor, anchor_emo, sample, sample_emo in batch:
            
            anchor, _ = librosa.load(anchor, sr=self.sr)
            sample, _ = librosa.load(sample, sr=self.sr)
            
            anchors.append(torch.tensor(anchor))
            anchors_emo.append(anchor_emo)
            samples.append(torch.tensor(sample))
            samples_emo.append(sample_emo)

            l_anchor_max = max(l_anchor_max, len(anchor))
            l_sample_max = max(l_sample_max, len(sample))
        
        y = []
        
        for idx in range(len(anchors)):
            
            ai = anchors[idx].size(0)
            pad = (0, l_anchor_max - ai)
            anchors[idx] = F.pad(anchors[idx], pad)
            
            si = samples[idx].size(0)
            pad = (0, l_sample_max - si)
            samples[idx] = F.pad(samples[idx], pad)
            
            y.append(min(abs(anchors_emo[idx] - samples_emo[idx]), 1))

        anchors = torch.stack(anchors)
        samples = torch.stack(samples)
        y = torch.tensor(y)
        
        return anchors, samples, y
    
    
class ESDatasetNPair(Dataset):
        
    def __init__(self, config):
        
        self.n_pairs = config['n_pairs']
        self.samples = []
        self.samples_emo = {'angry': [], 'happy': [], 'neutral': [], 'sad': [], 'surprise': []}
        
        for emo in config['dirs'].keys():
            
            emo_dir = config['dirs'][emo]
            if not os.path.isdir(emo_dir):
                raise FileNotFoundError(f'Cannot find the given directory: {emo_dir}')
            
            files = os.listdir(emo_dir)
            for f in files:
                fpath = os.path.join(emo_dir, f)
                if not os.path.isfile(fpath):
                    continue
                self.samples.append((fpath, emo))
                self.samples_emo[emo].append(fpath)
        
        self.loader = DataLoader(
            self, 
            batch_size=config.batch_size, 
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=Collate(config.sr),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        anchor, anchor_emo = self.samples[idx]
        positive = np.random.choice(self.samples_emo[anchor_emo])
        negative = []
        for emo in self.samples_emo.keys():
            if emo != anchor_emo:
                negative.append(np.random.choice(self.samples_emo[emo])) 
        # return: anchor, positive, negative
        return anchor, positive, negative

    
class CollateNPair:
    
    def __init__(self, sr):
        self.sr = sr
        
    def __call__(self, batch):
        
        l_max = 0
        samples = []

        for anchor, positive, negative in batch:
            
            anchor, _ = librosa.load(anchor, sr=self.sr)
            positive, _ = librosa.load(positive, sr=self.sr)
            
            negative_list, sig_len = [], []
            for neg in negative:
                neg, _ = librosa.load(neg, sr=self.sr)
                negative_list.append(torch.tensor(neg))
                sig_len.append(len(neg))
            l_max_i = max(sig_len + [len(anchor), len(positive)])
            l_max = max(l_max_i, l_max)
            
            sample = [torch.tensor(anchor), torch.tensor(positive)] + negative_list
            samples.append(sample)
        
        for sample_idx in range(len(samples)):
            for sig_idx in range(len(samples[sample_idx])):
                sig_len_i = samples[sample_idx][sig_idx].size(0)
                pad = (0, l_max - sig_len_i)
                samples[sample_idx][sig_idx] = F.pad(samples[sample_idx][sig_idx], pad)
            samples[sample_idx] = torch.stack(samples[sample_idx])
        samples = torch.stack(samples)
        
        return samples

