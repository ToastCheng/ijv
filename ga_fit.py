import sys 
import os 
from engine import Engine

from utils.ga import GA 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from glob import glob 
from copy import deepcopy 
import json 


wl = [i for i in range(660, 811, 10)]

if sys.argv[1] == "0524":
    date = "20190524_split"
    scvO2 = [0.7, 0.7, 0.7, 0.7, 0.7] + [0.7, 0.5, 0.7] * 3 + [0.7, 0.75, 0.7] * 3
    
elif sys.argv[1] == "0527":
    date = "20190527_split"
    scvO2 = [0.7] + [0.7, 0.5, 0.7] * 2 + [0.7, 0.75, 0.7] * 2 + [0.7, 0.6, 0.7] * 2
    
elif sys.argv[1] == "0528":
    date = "20190528_split"
    scvO2 = [0.7] + [0.7, 0.5, 0.7, 0.5, 0.7] * 2 + [0.7, 0.75, 0.7] + [0.7, 0.6, 0.7]
    
else:
    raise Exception("...?")

output_date = os.path.join("fitting", date)

if not os.path.isdir(output_date):
    os.mkdir(output_date)



spec_list = glob("data/calibrated/{}/IJV/{}_*_*.csv".format(date, date))
spec_list.sort(key=
    lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].strip('.csv')))
)

geo_max_path = "data/raw/{}/IJV/live/live_1_max.json".format(date)
geo_min_path = "data/raw/{}/IJV/live/live_1_min.json".format(date)
with open(geo_max_path) as f:
    g = json.load(f)
    geo_max = [
        g['skin_thickness'], 
        g['fat_thickness'], 
        g['ijv_radius'], 
        g['ijv_depth'], 
        g['cca_radius'], 
        g['cca_depth'], 
        g['ijv_cca_distance']
    ]
with open(geo_min_path) as f:
    g = json.load(f)
    geo_min = [
        g['skin_thickness'], 
        g['fat_thickness'], 
        g['ijv_radius'], 
        g['ijv_depth'], 
        g['cca_radius'], 
        g['cca_depth'], 
        g['ijv_cca_distance']
    ]

ga = GA(500, 24, 0, 0, geo_max, geo_min, iteration=501, dual_ratio=0.05, log_period=500)
ga.wavelength = wl
ga.engine.wl = wl



for s, scv in zip(spec_list, scvO2):
    print(s)
    idx = s.split(date)[-1].strip('.csv')[1:]
    
    output_dir = os.path.join(output_date, idx)
    output_fig = os.path.join(output_dir, "fig")
    output_log = os.path.join(output_dir, "log")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_fig):
        os.mkdir(output_fig)
    if not os.path.isdir(output_log):
        os.mkdir(output_log)
    
    
    
    print("="*10)
    spec = pd.read_csv(s)
    spec = spec[
        (spec["wavelength"] >= wl[0])&
        (spec["wavelength"] <= wl[-1])
    ]
    target_max = spec["max"].values.T
    target_min = spec["min"].values.T
    
    ga.scvO2 = scv
    ga.target_max = target_max
    ga.target_min = target_min
    ga.df = ga.df.iloc[0:0]

    ga(output_dir, plot=True, verbose=True)
    
    
