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

cosine = torch.nn.CosineSimilarity(dim=-1)

a = torch.rand(4, 768)
b = torch.rand(4, 768)
# c = torch.sum(torch.matmul(a, b.transpose(1, 0)))
# print(c)
# print(a.mean(dim=0).exp())
# c = cosine(a, b)
# print(c)
print(a)
print('-------------')
print(a.pow(2).sum(-1).sqrt())