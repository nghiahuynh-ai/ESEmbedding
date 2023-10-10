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
        mdist = torch.clamp(mdist, min=0.0)
        loss = (1 - y) * dist_sq + y * torch.pow(mdist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
    
    
class ContrastiveLossNPairs(nn.Module):
    
    def __init__(self, cluster_size):
        super(ContrastiveLossNPairs, self).__init__()
        self.cluster_size = cluster_size

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): (Sx5, 1, D)

        Returns:
            tensor: (1)
        """
        
        x = x.squeeze(1)
        pos, centroids = 0, []

        for idx in range(5):
            
            start = idx * self.cluster_size
            end = start + self.cluster_size
            cluster = x[start: end]
            scalar = torch.matmul(cluster, cluster.transpose(1, 0))
            print('----------------------------------------------')
            print(cluster.pow(2).sum(-1).sqrt().shape)
            print('----------------------------------------------')
            print(cluster.transpose(1, 0).pow(2).sum(-1).sqrt().shape)
            length = torch.matmul(cluster.pow(2).sum(-1).sqrt(), cluster.transpose(1, 0).pow(2).sum(-1).sqrt())
            score = torch.sum(torch.exp(scalar / length))
            pos += score
            
            centroids.append(cluster.mean(dim=0))
            
        centroids = torch.stack(centroids).to(x.device)
        scalar = torch.matmul(centroids, centroids.transpose(1, 0))
        length = torch.matmul(centroids.pow(2).sum(-1).sqrt(), centroids.transpose(1, 0).pow(2).sum(-1).sqrt())
        neg = torch.sum(torch.exp(scalar / length))
        
        loss = -torch.log(pos / neg)
        
        return loss