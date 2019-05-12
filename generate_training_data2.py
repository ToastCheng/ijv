import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from MySQLdb import connect

import os 
import json
import random
import string
from time import time
from collections import defaultdict

from utils.mch import MCHHandler


x_range = {
    "idx": None,
    "skin_b": (0.0012, 0.0041),
    "skin_s": (0.985, 1.0),
    "skin_w": (0.166, 0.261),
    "skin_f": (0.187, 0.277),
    "skin_m": (0.0087, 0.0165),

    "fat_f": (0.5, 1),

    "muscle_b": (0.005, 0.050),
    "muscle_s": (0.5, 0.9),
    "muscle_w": (0.5, 0.9),

    "ijv_s": (0.0, 1.0),

    "cca_s": (0.0, 1.0),
}


def get_idx(n=10):
    assert n > 1, "n should be greater than 1"
    first = random.choice(string.ascii_letters)
    rest = "".join(random.sample(string.ascii_letters + string.digits, n-1))
    return first + rest



df = pd.read_csv(os.path.join("train", "data_list.csv"))
# x = {i: [] for i in x_range.keys()}
x = defaultdict(list)
mch = MCHHandler()

for step_one_idx in range(1):
    
    ### 準備好mch的結果
    with open(os.path.join("configs", "train.json"), "r") as f:
        config = json.load(f)
    wl = pd.read_csv(config["wavelength"])["wavelength"].values
    session_id = df["idx"][0]
    config["session_id"] = session_id
    config["parameters"] = os.path.join("train", "input", session_id + ".json")

    path = os.path.join("configs", "train", session_id + ".json")
    with open(path, "w+") as f:
        json.dump(config, f, indent=4)
        
    mch.load_config(path)
    
    ###
    
    ### random生產100組吸收係數，並存下來
    conn = connect(
        host="140.112.174.26",
        user="md703",
        passwd=os.getenv("PASSWD"),
        db="ijv"
    )
    for i in tqdm(range(100)):
        for xx in x_range.keys():
            if xx == "idx":
                x[xx] += [get_idx()]            
            else:
                x[xx] += [random.uniform(x_range[xx][0], x_range[xx][1])]
        for xx in df.columns[1:]:
            x[xx] += [df[xx][step_one_idx]]
        
        sql = "INSERT INTO ijv_sim_spec(idx, skin_b, skin_s, skin_w, skin_f, skin_m, \
        fat_f,muscle_b, muscle_s, muscle_w, \
        ijv_s, cca_s, \
        skin_musp, skin_bmie, fat_musp, fat_bmie, \
        muscle_musp, muscle_bmie,ijv_musp', cca_musp, \
        geo_skin, geo_fat, geo_ijvr, geo_ijvd,\
        geo_ccar, geo_ccad, geo_ijvcca) \
        VALUES('%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', \
        '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', \
        '%f', '%f', '%f', '%f')" % \
        (x["idx"][i], x["skin_b"][i], x["skin_s"][i], x["skin_w"][i], x["skin_f"][i], x["skin_m"][i],\
         x["fat_f"][i], x["muscle_b"][i], x["muscle_s"][i], x["muscle_w"][i], \
         x["ijv_s"][i], x["cca_s"][i], \
         x["skin_musp"][i], x["skin_bmie"][i], x["fat_musp"][i], x["fat_bmie"][i], \
         x["muscle_musp"][i], x["muscle_bmie"][i], x["ijv_musp"][i], x["cca_musp"][i], \
         x["geo_skin"][i], x["geo_fat"][i], x["geo_ijvr"][i], x["geo_ijvd"][i], \
         x["geo_ccar"][i], x["geo_ccad"][i], x["geo_ijvcca"][i])

        ijv_args = {
            "skin":{
                "blood_volume_fraction": x["skin_b"][i],
                "ScvO2": x["skin_s"][i],
                "water_volume":x["skin_w"][i],
                "fat_volume": x["skin_f"][i],
                "melanin_volume": x["skin_m"][i],
            },

            "fat":{
                "blood_volume_fraction": 0,
                "ScvO2": 0,
                "water_volume": 0,
                "fat_volume": x["fat_f"][i],
                "melanin_volume": 0,
            },

            "muscle":{
                "blood_volume_fraction": x["muscle_b"][i],
                "ScvO2": x["muscle_s"][i],
                "water_volume": x["muscle_w"][i],
                "fat_volume": 0,
                "melanin_volume": 0,
            },
            "ijv":{
                "blood_volume_fraction": 1,
                "ScvO2": x["ijv_s"][i],
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
            },
            "cca":{
                "blood_volume_fraction": 1,
                "ScvO2": x["cca_s"][i],
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
            },
        }
    
        
        start = time()
        s, p = mch.run_wmc(ijv_args)
        end = time()
        # print("time: ", end - start)
        if not os.path.isdir(os.path.join("train", "spec", x["idx"][i])):
            os.mkdir(os.path.join("train", "spec", x["idx"][i]))
        np.save(os.path.join("train", "spec", x["idx"][i], x["idx"][i]), s)
        
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()

        plt.figure()
        for n, ss in enumerate(s[5:]):
            plt.plot(wl, ss, label="SDS: %d" % n)
        plt.legend()
        plt.grid()
        plt.xlabel("wavelength [nm]")
        plt.ylabel("reflectance [-]")
        plt.savefig(os.path.join("train", "spec", x["idx"][i], x["idx"][i] + ".png"))
        plt.clf()
        
df_full = pd.DataFrame(x)
df_full.to_csv(os.path.join("train", get_idx()+".csv", index=None))