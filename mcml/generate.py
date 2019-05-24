import os 
import json
import random
import string
from time import time



def get_idx(n=10):
    assert n > 1, "n should be greater than 1"
    first = random.choice(string.ascii_letters)
    rest = "".join(random.sample(string.ascii_letters + string.digits, n-1))
    return first + rest

def get_param():
    upper_thickness = random.uniform(0, 0.3)
    up_mua = random.uniform(0, 20)
    up_mus = random.uniform(0, 500)
    up_tissue_n = random.uniform(1.3, 1.5)
    up_g = random.uniform(0, 1)

    mid_thickness = random.uniform(0, 0.3)
    mid_mua = random.uniform(0, 20)
    mid_mus = random.uniform(0, 500)
    mid_tissue_n = random.uniform(1.3, 1.5)
    mid_g = random.uniform(0, 1)

    third_mua = random.uniform(0, 20)
    third_mus = random.uniform(0, 500)
    third_tissue_n = random.uniform(1.3, 1.5)
    third_g = random.uniform(0, 1)

    return (upper_thickness, up_mua, up_mus, up_tissue_n, up_g,
        mid_thickness, mid_mua, mid_mus, mid_tissue_n, mid_g,
        third_mua, third_mus, third_tissue_n, third_g)


for i in range(10000):
    idx = get_idx()
    input_path = os.path.join("input", idx)
    output_path = os.path.join("output", idx)
    with open(idx, "w+") as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(*get_param()))
    os.system("./mc {} {}".format(input_path, output_path))
    


