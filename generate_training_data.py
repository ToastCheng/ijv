import pandas as pd 
import numpy as np 
import string
import random 
import json

x_col = [
    "idx",
    "skin_musp",
    "skin_bmie",
    "fat_musp",
    "fat_bmie",
    "muscle_musp",
    "muscle_bmie",
    "ijv_musp",
    "cca_musp",
    "geo_skin",
    "geo_fat",
    "geo_ijvr",
    "geo_ijvd",
    "geo_ccar",
    "geo_ccad",
    "geo_ijvcca"
]

# 1/cm
x_range = {
    "skin_musp": (29.7, 48.9),
    "skin_bmie": (0.705, 2.453),
    "fat_musp": (13.7, 35.8),
    "fat_bmie": (0.385, 0.988),
    "muscle_musp": (9.8, 13.0),
    "muscle_bmie": (0.926, 2.82),
    "ijv_musp": (10, 20),
    "cca_musp": (10, 20),
# mm
    "geo_skin": (0.5, 1.5),
    "geo_fat": (0.5, 1.5),
    "geo_ijvr": (2.5, 10.5),
    "geo_ijvd": (2, 16),
    "geo_ccar": (2.5, 4.5),
    "geo_ccad": (5, 26),
    "geo_ijvcca": (5, 10)
}

def get_idx(n=10):
    assert n > 1, "n should be greater than 1"
    first = random.choice(string.ascii_letters)
    rest = "".join(random.sample(string.ascii_letters + string.digits, n-1))
    return first + rest


x = {i: [] for i in x_col}
for i in range(100):
    for xx in x_col[:-7]:
        if xx == "idx":
            x[xx] += [get_idx()]            
        else:
            x[xx] += [random.uniform(x_range[xx][0], x_range[xx][1])]
    ["geo_skin", "geo_fat", "geo_ijvr", "geo_ijvd", "geo_ccar", "geo_ccad", "geo_ijvcca"]          
    x["geo_skin"] += [random.uniform(x_range["geo_skin"][0], x_range["geo_skin"][1])]
    x["geo_fat"] += [random.uniform(x_range["geo_fat"][0], x_range["geo_fat"][1])]
    x["geo_ijvd"] += [random.uniform(x_range["geo_ijvd"][0], x_range["geo_ijvd"][1])]
    x["geo_ijvr"] += [random.uniform(x_range["geo_ijvr"][0], x_range["geo_ijvr"][1])]
    while x["geo_skin"][-1] + x["geo_fat"][-1] > x["geo_ijvd"][-1] - x["geo_ijvr"][-1]:
        x["geo_ijvd"][-1] = random.uniform(x_range["geo_ijvd"][0], x_range["geo_ijvd"][1])
        x["geo_ijvr"][-1] = random.uniform(x_range["geo_ijvr"][0], x_range["geo_ijvr"][1])
    
    x["geo_ccad"] += [random.uniform(x_range["geo_ccad"][0], x_range["geo_ccad"][1])]
    x["geo_ccar"] += [random.uniform(x_range["geo_ccar"][0], x_range["geo_ccar"][1])]
    x["geo_ijvcca"] += [random.uniform(x_range["geo_ijvcca"][0], x_range["geo_ijvcca"][1])]
    while (x["geo_ccad"][-1] - x["geo_ijvd"][-1])**2 + (x["geo_ijvcca"][-1])**2 <  (x["geo_ccar"][-1] + x["geo_ijvr"][-1])**2:
        x["geo_ccad"][-1] = random.uniform(x_range["geo_ccad"][0], x_range["geo_ccad"][1])
        x["geo_ccar"][-1] = random.uniform(x_range["geo_ccar"][0], x_range["geo_ccar"][1])
        x["geo_ijvcca"][-1] = random.uniform(x_range["geo_ijvcca"][0], x_range["geo_ijvcca"][1])
    
    
df = pd.DataFrame(x)
df.to_csv("train/data_list.csv", index=None)

with open("train/input/template.json") as f:
    inp = json.load(f)

for i in range(len(df)):
    inp["idx"] = df["idx"][i]
    inp["skin"]["muspx"] = df["skin_musp"][i]
    inp["skin"]["bmie"] = df["skin_bmie"][i]

    inp["fat"]["muspx"] = df["fat_musp"][i]
    inp["fat"]["bmie"] = df["fat_bmie"][i]

    inp["muscle"]["muspx"] = df["muscle_musp"][i]
    inp["muscle"]["bmie"] = df["muscle_bmie"][i]

    inp["IJV"]["muspx"] = df["ijv_musp"][i]
    inp["IJV"]["bmie"] = 1.0

    inp["CCA"]["muspx"] = df["cca_musp"][i]
    inp["CCA"]["bmie"] = 1.0

    inp["geometry"]["skin_thickness"] = df["geo_skin"][i]
    inp["geometry"]["fat_thickness"] = df["geo_fat"][i]
    inp["geometry"]["ijv_radius"] = df["geo_ijvr"][i]
    inp["geometry"]["ijv_depth"] = df["geo_ijvd"][i]
    inp["geometry"]["cca_radius"] = df["geo_ccar"][i]
    inp["geometry"]["cca_depth"] = df["geo_ccad"][i]
    inp["geometry"]["ijv_cca_distance"] = df["geo_ijvcca"][i]
    inp["boundary"]["x_size"] = 150
    inp["boundary"]["y_size"] = 150
    inp["boundary"]["z_size"] = 200
    

    with open("train/input/" + df["idx"][i] + ".json", "w+") as f:
        json.dump(inp, f, indent=4)

