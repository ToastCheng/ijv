import os
import json 
from glob import glob

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 

import utils
from mcx import MCX
from generator import Generator


def main():
	test_whiteMC()
	test_smooth()

def test_whiteMC():
	"""
	This function is writen to test whether white Monte Carlo is 
	adequate for the IJV tissue model
	"""

	# white
	mcx = MCX("test/test_white.json")
	mcx.run()
	mcx.calculate_reflectance(plot=False)

	reflec_white = mcx.reflectance[70][-1]

	# normal
	mcx2 = MCX("test/test_white.json")
	mcx2.run(white=False)
	mcx2.calculate_reflectance(white=False, plot=False)

	reflec = mcx2.reflectance[-1]


	plt.plot(reflec, label='normal')
	plt.plot(reflec_white, label='white')
	plt.legend()
	plt.savefig("test/whiteMC.png")
	plt.close()

	print(rmse(reflec_white, reflec))

	df = pd.DataFrame({"white": reflec_white, "normal": reflec})
	df.to_csv("test/result_whiteMC.csv", index=False)


def test_smooth():
	"""
	This function is writen to test whether we can use smooth technique
	to lower the number of photon and maintain the spectrum shape,
	in that, we can speed up the simulation
	"""
	mcx = MCX('test/test_smooth_2e8.json')
	mcx.run()
	mcx.calculate_reflectance()

	mcx2 = MCX('test/test_smooth_2e9.json')
	mcx2.run()
	mcx2.calculate_reflectance()



def rmse(x, y):
	e = np.sqrt(((x - y)**2).mean())
	return e/y.mean()



if __name__ == "__main__":
	main()