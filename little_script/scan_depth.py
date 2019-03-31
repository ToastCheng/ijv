import os
import json 
from glob import glob
from argparse import ArgumentParser

from utils.line import LineBot
from mcx import MCX
from generator import Generator


line = LineBot()


def scan_depth():
    depth = [8, 10, 12, 14, 16, 18, 20]
    configs = "configs/scan_depth.json"

    for idx, d in enumerate(depth):

        with open(configs) as f:
            c = json.load(f)

        c["session_id"] = "20190318_scan_depth_%d" % d
        c["input_file"] = "input/parameters_20190318_%d.json" % idx

        with open(configs, "w+") as f:
            json.dump(c, f, indent=4)
        
        mcx = MCX(configs)
        mcx.run()
        mcx.calculate_reflectance()
        line.print("跑完depth: %dmm" % d)


if __name__ == "__main__":
    scan_depth()