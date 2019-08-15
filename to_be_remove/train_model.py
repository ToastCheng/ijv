import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from MySQLdb import connect


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_layer = nn.Linear(29, 64)
        self.geo_layer = nn.Linear(7, 64)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, spec, geo):
        spec = self.spec_layer(spec)
        spec = torch.relu(spec)
        
        geo = self.geo_layer(geo)
        geo = torch.relu(geo)
        
        out = self.fc1(spec + geo)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        
        return out


class SpecData(Dataset):
    def __init__(self, df):
        self.df = df
        

    def __getitem__(self, idx):
        i = self.df["idx"][idx]
        path = "train/spec/{}/{}.npy".format(i, i)
        spec = np.load(path)[5] # SDS: 20mm 
        spec = torch.tensor(spec).float()
        
        geo = self.df[["geo_skin", "geo_fat", "geo_ijvr", "geo_ijvd", "geo_ccar", "geo_ccad", "geo_ijvcca"]].values[idx]
        geo = torch.tensor(geo).float()
        
#         param = self.df[['skin_b', 'skin_s', 'skin_w', 'skin_f', 'skin_m', 'fat_f',
#        'muscle_b', 'muscle_s', 'muscle_w', 'ijv_s', 'cca_s', 'skin_musp',
#        'skin_bmie', 'fat_musp', 'fat_bmie', 'muscle_musp', 'muscle_bmie',
#        'ijv_musp', 'cca_musp']].values[idx]
        param = self.df['ijv_s'].values[idx]
        param = torch.tensor(param).float()
        return spec, geo, param
    
    def __len__(self):
        return len(self.df)



model = Model()
# model.to(device)
loss_func = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(lr=1e-4, params=model.parameters())

conn = connect(
    host="140.112.174.26",
    user="md703",
    passwd="MD703",
    db="ijv"
)
df = pd.read_sql("SELECT * FROM ijv_sim_spec", con=conn)

dataset = SpecData(df)
dataloader = DataLoader(dataset, batch_size=8,
                        shuffle=True, num_workers=1)


for epoch in range(1000):
    print("epoch: ", epoch, end="\r")
    for i, (spec, geo, param) in enumerate(dataloader):
#         spec, geo, param = spec.to(device), geo.to(device), param.to(device)
        optimizer.zero_grad()
        predict = model(spec, geo)
        loss = loss_func(predict, param)
        loss.backward()
        optimizer.step()
    # loss_list += [float(loss.data)]
    # lr_list += [optimizer.param_groups[0]["lr"]]
    if epoch % 10 == 0:
        print("loss: ", float(loss.data))

