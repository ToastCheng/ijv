import os 
import json
import random
import string
from time import time


for pid in "chik":
    input_path = os.path.join("input", "{}.txt".format(pid))
    output_path = os.path.join("output", "{}.txt".format(pid))
    
    os.system("./mc {} {}".format(input_path, output_path))



