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


config = OmegaConf.load('config.yaml')
print(config)
# model = ESEmbedding(config)

# sig = np.random.rand(16, 16000)
# out = model(sig)
# print(out.shape)