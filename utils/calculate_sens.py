import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt

def cal_sen(x1, x2):
    return np.abs(x2 - x1).mean()/(x1).mean()

def cal_sen2(x1, x2):
    return np.abs(x2 - x1)/x1

def calculate_sens(self, data):
	# data -> [ScvO2, SDS, wl]

	# 1
	percentage = [i for i in range(100)]
	wl = [str(i) for i in range(650, 1001, 10)]

	sensitivity = []
	num_sds = data.shape[1]
	num_wl = data.shape[2]

	plt.figure(figsize=(12,8))
	for s in range(num_sds):
	    sen = []
	    for i in range(100):
	        ss = cal_sen(data[i, s, :], data[i+1, s, :])/1
	        sen.append(ss)

	    sensitivity.append(sen)
	    plt.plot(percentage, sen, label="sds %d" % s)

	plt.grid()
	plt.legend()
	plt.title("sensitivity on SDS #%d" % s)
	plt.ylabel("sensitivity")
	plt.xlabel("ScvO2")

	plt.savefig("sens_%d.png" % s)
	plt.clf()

	# 2
	plt.figure(figsize=(12,8))
	sens = np.asarray(sensitivity)

	plt.imshow(sens, aspect='auto')
	plt.xlabel("ScvO2")
	plt.ylabel("SDS")
	plt.colorbar()
	plt.savefig("new_hm.png")
	plt.clf()

	# 3


	for s in range(num_sds):
	    sww = []
	    for p in range(100):
	        sw = []
	        for w in range(36):
	            sw.append(cal_sen2(data[0i, s, w], data[i+1, s, w]))
	        sww.append(sw)
	    sww = np.asarray(sww)
	    plt.figure(figsize=(18, 10))
	    plt.imshow(sww, aspect='auto')
	    plt.xticks([i for i in range(36)], labels=wl)
	    plt.colorbar()
	    plt.xlabel("wavelength")
	    plt.ylabel("ScvO2")
	    plt.title("new fiber SDS #%d" % s)
	#     plt.show()
	    plt.savefig("wl_hm_%d.png" % s)
	    plt.clf()
    

	# 4

	plt.figure(figsize=(16, 6))

	for s in range(num_sds):
	    sww = []
	    for p in range(100):
	        sw = []
	        for w in range(36):
	            sw.append(cal_sen2(data[i, s, w], data[i+1, s, w]))
	        sww.append(sw)
	    sww = np.asarray(sww)
	    plt.plot(sww.mean(0), label="SDS #%d" % s)
	plt.legend()
	plt.grid()
	plt.xlabel("wavelength")
	plt.xticks([i for i in range(35)], [str(i) for i in range(650, 1001, 10)])
	plt.ylabel("sensitivity")
	plt.savefig("sens_wl.png")