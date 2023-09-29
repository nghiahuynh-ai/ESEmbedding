import torch
import torch.nn as nn
from einops import repeat

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
    
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = (1 - y) * dist_sq + y * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
    
    
class ContrastiveLossNPairs(nn.Module):
    
    def __init__(self):
        super(ContrastiveLossNPairs, self).__init__()

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): (B, N+1 pairs, D)

        Returns:
            tensor: (B)
        """
        
        anchor, positive, negative = x[:, 0, :], x[:, 1, :], x[:, 2:, :]
        sim = anchor - positive
        sim = torch.sum(torch.pow(sim, 2), 1)
        sim = torch.sqrt(sim)

        anchor = anchor.unsqueeze(1)
        anchor = repeat(anchor, 'b 1 d -> b n d', n=negative.size(1))
        unsim = anchor - negative
        unsim = torch.sum(torch.pow(unsim, 2), 2)
        unsim = torch.sum(torch.sqrt(unsim), 1)
        
        loss = -torch.log(sim / unsim)
        return loss.mean()