from glob import glob
import pandas as pd 
import json
import os 
from mcx import MCX

train_list = pd.read_csv("train/data_list.csv")

mcx = MCX()

for i, idx in enumerate(train_list["idx"]):
    with open("configs/train.json") as f:
        config = json.load(f)

    config["session_id"] = idx
    config["parameters"] = os.path.join("train", "input", idx + ".json")

    with open("configs/train.json") as f:
        json.dump(config, f, indent=4)

    print("run: ", i)
    mcx.run(config_file="configs/train.json")

