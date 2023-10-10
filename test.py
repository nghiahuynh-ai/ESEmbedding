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

loss = ContrastiveLossNPairs(4)

a = torch.rand(20, 1, 768)
out = loss(a)
print(out)