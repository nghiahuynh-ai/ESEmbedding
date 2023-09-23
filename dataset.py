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
        sample = np.random.choice(self.samples)
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
            anchors_emo.append(torch.tensor([anchor_emo]))
            samples.append(torch.tensor(sample))
            samples_emo.append(torch.tensor([sample_emo]))

            l_anchor_max = max(l_anchor_max, len(anchor))
            l_sample_max = max(l_sample_max, len(sample))
        
        for idx in range(len(anchors)):
            
            ai = anchors[idx].size(2)
            pad = (0, l_anchor_max - ai)
            anchors[idx] = F.pad(anchors[idx], pad)
            
            si = samples[idx].size(2)
            pad = (0, l_sample_max - si)
            samples[idx] = F.pad(samples[idx], pad)

        anchors = torch.stack(anchors)
        anchors_emo = torch.stack(anchors_emo)
        samples = torch.stack(samples)
        samples_emo = torch.stack(samples_emo)
        
        return anchors, anchors_emo, samples, samples_emo

