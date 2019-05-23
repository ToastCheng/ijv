import os
from glob import glob

file_list = glob("*.mch")

for f in file_list:
	f = f.split("/")[-1]
	print(f)
	date, type_, wl = f.split('_')
	
	os.system("mv {}_{}_{} {}_{}_no_prism_{}".format(date, type_, wl, date, type_, wl))

