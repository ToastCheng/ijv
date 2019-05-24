import os 
import sys
from time import time
from random import uniform

import pandas as pd 
import numpy  as np 
import torch
import matplotlib.pyplot as plt 

from tqdm import tqdm

from __init__ import load_mch
from mch import MCHHandler

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
wl = [i for i in range(660, 921, 10)]

x_range = {
    "sb": (0, 1.0),
    "ss": (0, 1.0),
    "sw": (0, 0.5),
    "sf": (0, 0.5),
    "sm": (0, 1.0),

    "ff": (1, 1),

    "mb": (0.005, 0.1),
    "ms": (0.0, 0.2),
    "mw": (0.0, 0.3),

    "is": (0.4, 0.8),

    "cs": (0.5, 1.0),
}

def get_args(xrange, num=50000):
    args = []
    for i in range(num):
        sb = uniform(x_range["sb"][0], x_range["sb"][1])
        ss = uniform(x_range["ss"][0], x_range["ss"][1])
        sw = uniform(x_range["sw"][0], max(x_range["sw"][0], min(x_range["sw"][1], 1-sb)))
        sf = uniform(x_range["sf"][0], max(x_range["sf"][0], min(x_range["sf"][1], 1-sb-sw)))
        sm = 1-sb-sw-sf
        mb = uniform(x_range["mb"][0], x_range["mb"][1])
        ms = uniform(x_range["ms"][0], x_range["ms"][1])
        mw = uniform(x_range["mw"][0], max(x_range["mw"][0], min(x_range["mw"][1], 1-mb)))
        mc = 1 - mb - mw
        is_ = uniform(x_range["is"][0], x_range["is"][1])
        cs = uniform(x_range["cs"][0], x_range["cs"][1])
        arg = {
            "skin":{
                "blood_volume_fraction": sb,
                "ScvO2": ss,
                "water_volume": sw,
                "fat_volume": sf,
                "melanin_volume": sm,
                "collagen_colume": 0
            },

            "fat":{
                "blood_volume_fraction": 0,
                "ScvO2": 0,
                "water_volume": 0,
                "fat_volume": 1,
                "melanin_volume": 0,
                "collagen_colume": 0
            },

            "muscle":{
                "blood_volume_fraction": mb,
                "ScvO2": ms,
                "water_volume": mw,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_colume": mc
            },
            "ijv":{
                "blood_volume_fraction": 1,
                "ScvO2": is_,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_colume": 0
            },
            "cca":{
                "blood_volume_fraction": 1,
                "ScvO2": cs,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_colume": 0
            },
        }
        args += [arg]
    
    return args

args = get_args(x_range)


date = sys.argv[1]
idx = os.path.join("configs", "live", date)
max_no_prism = idx + "_max" + "_no_prism" + ".json"
max_prism = idx + "_max" + ".json"
min_no_prism = idx + "_min" + "_no_prism" + ".json"
min_prism = idx + "_min" + ".json"

# 跑WMC!!

mch = MCHHandler()

mch.load_config(max_no_prism)
s_max, r_max = mch.run_wmc(args)

mch.load_config(min_no_prism)
s_min, r_min = mch.run_wmc(args)


# 活體光譜

live_list = glob("data/calibrated/{}/IJV/{}*.csv".format(date, date))


for ll in live_list:
    df = pd.read_csv(ll)

    loss = 100
    for i, s in enumerate(s_min):
        if np.sqrt((s[1,:]-df["max"])**2).mean()/df["max"].mean() < loss:
            loss = np.sqrt((s[1, :] - df["max"])**2).mean()/df["max"].mean()
            print("num: {} | loss: {}".format(i, loss))

            plt.plot(wl, s[1, :], label="simulated spectra") 
            plt.plot(wl, df["max"], label="measured spectra")
            plt.xlabel("wavelength [nm]")
            plt.ylabel("reflectance [-]")
            plt.legend()
            plt.grid()
            plt.show()

    loss = 100
    for i, s in enumerate(s_max):
        if np.sqrt((s[1,:]-df["min"])**2).mean()/df["min"].mean() < loss:
            
            loss = np.sqrt((s[1, :] - df["min"])**2).mean()/df["min"].mean()
            print("num: {} | loss: {}".format(i, loss))
            
            plt.plot(wl, s[1, :], label="simulated spectra") 
            plt.plot(wl, df["min"], label="measured spectra")
            plt.xlabel("wavelength [nm]")
            plt.ylabel("reflectance [-]")
            plt.legend()
            plt.grid()
            plt.show()



