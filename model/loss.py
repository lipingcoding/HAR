import pdb

import torch
import torch.nn as nn



class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bceloss = nn.BCELoss(reduction='none')


    def forward(self, p, y, mask):
        loss =  self.bceloss(p, y)
        loss = mask.unsqueeze(-1) * loss
        loss = loss.sum() / mask.sum()
  
        return loss


