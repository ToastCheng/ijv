import torch
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_layer = nn.Linear(20, 64)
        self.bn_spec = nn.BatchNorm1d(64)
        self.geo_layer = nn.Linear(7, 64)
        self.bn_geo = nn.BatchNorm1d(64)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, spec, geo):
        spec = self.spec_layer(spec)
        spec = self.bn_spec(spec)
        spec = nn.ELU()(spec)
        
        geo = self.geo_layer(geo)
        geo = self.bn_geo(geo)
        geo = nn.ELU()(geo)
        
        out = self.fc(spec + geo)
       
        return out
