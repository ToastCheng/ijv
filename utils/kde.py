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


class KDE:
	def __init__(self): 
		pass

	def fit(self, x, band_width=3):
		smoothed = np.zeros(len(x))
		for idx, xx in enumerate(x):
			smoothed += xx*self.gaussian(np.arange(len(x)), idx, band_width)
		
		return smoothed 

	@staticmethod
	def gaussian(x, mean, std):

		x = np.asarray(x)
		denominator = std*np.sqrt(2*np.pi)
		numerator = np.exp(-0.5*((x-mean)/std)**2)
		distribution =  numerator/denominator
		distribution /= distribution.sum()

		return distribution


if __name__ == "__main__":
	import matplotlib.pyplot as plt 

	x = np.random.random(100) * np.sin(np.arange(100) * 0.01 * np.pi)
	kde = KDE()

	for i in range(1, 10, 2):
		s = kde.fit(x, i)


		plt.title('Kernel Density Estimation')
		plt.plot(x, label='raw')
		plt.plot(s, label='smoothed')
		plt.xlabel('band width: %d' % i)
		plt.legend()
		plt.savefig('test_kde/std_%d.png' % i)
		plt.clf()