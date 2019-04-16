import numpy as np 


def smooth(x, band_width=3):
	def gaussian(x, mean, std):

		x = np.asarray(x)
		denominator = std*np.sqrt(2*np.pi)
		numerator = np.exp(-0.5*((x-mean)/std)**2)
		distribution =  numerator/denominator
		distribution /= distribution.sum()

		return distribution
	
	smoothed = np.zeros(len(x))
	for idx, xx in enumerate(x):
		smoothed += xx*gaussian(np.arange(len(x)), idx, band_width)
	
	return smoothed 