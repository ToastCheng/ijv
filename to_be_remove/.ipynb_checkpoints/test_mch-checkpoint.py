from utils.mch import MCHHandler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt




mch = MCHHandler()
mch.load_config("configs/train.json")

args = {
    "skin":{
        "blood_volume_fraction": 0.5,
        "ScvO2": 0.7,
        "water_volume": 0.49,
        "fat_volume": 0,
        "melanin_volume": 0.01,
    },

    "fat":{
        "blood_volume_fraction": 0,
        "ScvO2": 0,
        "water_volume": 0,
        "fat_volume": 1,
        "melanin_volume": 0,
    },

    "muscle":{
        "blood_volume_fraction": 0.05,
        "ScvO2": 0.80,
        "water_volume": 0.75,
        "fat_volume": 0,
        "melanin_volume": 0,
    },
    
    "IJV": {
        "ScvO2": 0.7
    },
    "CCA": {
        "ScvO2": 0.7
    }
}

s, p = mch.run_wmc(args)

