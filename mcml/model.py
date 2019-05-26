import torch
import torch.nn as nn




class Muscle(nn.Module):
    def __init__(self):
        super(Muscle, self).__init__()

    def forward(self, x):
        return x