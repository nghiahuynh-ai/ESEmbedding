from omegaconf import OmegaConf
import torch
import numpy as np
from transformers import (
    AutoModel,
    DebertaV2Model,
    Wav2Vec2Model,
    AutoProcessor,
)
from model_core import ESEmbedding
from loss import ContrastiveLoss, ContrastiveLossNPairs


processor = AutoProcessor.from_pretrained('facebook/wav2vec2-base')
sig = torch.rand(2, 6, 1024)
processed = processor(sig, sampling_rate=16000, return_tensors='pt')
print(processed.get('input_values').shape)