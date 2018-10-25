import json
from time import time
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_mch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MCHHandler:
	# This class is a .mch file handler.

	def __init__(
		self,
		mch_file=None, 
		config="config.json"
		):

		with open(config) as f:
			self.config = json.loads(f.read())
		if mch_file is not None:
			self.df, self.header, self.photon = self._load_mch(mch_file)
		else:
			self.df = None
			self.header = None
			self.photon = None
		self.mua = pd.read_csv(self.config["absorb_file"])
		self.wavelength = pd.read_csv(self.config["wavelength_file"])
		with open(self.config["input_file"]) as f: 
			self.input = json.loads(f.read())

		self.detector_na = 0.22
		self.detector_n = 1.457
		# self.critical_angle = np.arcsin(self.detector_na/self.detector_n)
		self.critical_angle = 12


	def compute_reflectance_white(self, mch_file=None, test=False):
		# reload a .mch file
		if mch_file is not None:
			self.df, self.header, self.photon = self._load_mch(mch_file)

		df = self._parse_mch()
		# filter the photon that not been detected(due to critical angle)
		df = df[np.arccos(df.angle.abs()) <= self.critical_angle]
		if len(df) == 0:
			print('no photon detected')
			return
		df = df.reset_index(drop=True)


		# test speed
		if test == True:
			num_photon = int(1e6)
			idx = np.random.randint(4, size=(num_photon, 1)) + 1
			path = np.random.rand(num_photon, 3)
			angle = np.random.rand(num_photon, 1)
			data = np.concatenate([idx, path, angle], 1)

			df = pd.DataFrame(data, columns=['detector_idx', 'media_0', 'media_1', 'media_2', 'angle'])


		# [medium, wavelength, ScvO2]
		# ->[ScvO2, wavelength, medium]
		mua = self._make_tissue_white()
		mua = torch.tensor(mua).float().to(device)

		# [medium, photon]
		path_length = df.iloc[:, 1:-1].values
		path_length = torch.tensor(path_length).transpose(1, 0).float().to(device)

		# [ScvO2, wavelength, photon]
		# print(-torch.matmul(mua, path_length)*self.header["unitmm"])
		weight = torch.exp(-torch.matmul(mua, path_length)*self.header["unitmm"]) 
		
		# [ScvO2, wavelength, 1]
		result = torch.zeros(weight.shape[:-1]).unsqueeze(2).float().to(device)

		# seperate photon with different detector
		for idx in range(1, self.header["detnum"]+1):

			# get the index of specific index
			detector_list = df.index[df["detector_idx"] == idx].tolist()

			# pick the photon that detected by specific detector
			# [ScvO2, wavelength, 1]
			# print("go")
			# print(result.shape)
			# print(weight.shape)
			_weight = weight[:, :, detector_list]
			# print(_weight.shape)
			_weight = _weight.sum(2)
			_weight = _weight.unsqueeze(2)
			# print(_weight.shape)
			result = torch.cat((result, _weight), 2)

		# [ScvO2, wavelength, SDS]
		result = result[:, :, 1:]

		return result.cpu().numpy()/self.header["total_photon"]
		
	def _make_tissue_white(self):

		# the ScvO2
		ScvO2 = np.arange(0, 1.01, 0.01)
		
		# mua
		oxy = self.mua['oxy'].values
		deoxy = self.mua['deoxy'].values
		water = self.mua['water'].values
		wl = self.mua['nmlib'].values

		# interpolation
		oxy = np.interp(self.wavelength, wl, oxy)
		deoxy = np.interp(self.wavelength, wl, deoxy)
		water = np.interp(self.wavelength, wl, water)

		# turn the unit 1/cm --> 1/mm
		oxy *= 0.1
		deoxy *= 0.1
		water *= 0.1

		# [medium, wavelength, 1]
		mua = np.zeros((3, len(self.wavelength), 1))


		for s in ScvO2:

			muscle = self._calculate_mua(
				self.input["muscle"]["blood_volume_fraction"],
				self.input["muscle"]["ScvO2"],
				self.input["muscle"]["water_volume"],
				oxy, 
				deoxy, 
				water
				)

			IJV = self._calculate_mua(
				self.input["IJV"]["blood_volume_fraction"],
				s,	# set ScvO2
				self.input["IJV"]["water_volume"],
				oxy, 
				deoxy, 
				water
				)
			CCA = self._calculate_mua(
				self.input["CCA"]["blood_volume_fraction"],
				self.input["CCA"]["ScvO2"],
				self.input["CCA"]["water_volume"],
				oxy, 
				deoxy, 
				water
				)

			_mua = np.concatenate(
				[np.expand_dims(muscle, 0), 
				 np.expand_dims(IJV, 0), 
				 np.expand_dims(CCA, 0)], 0
				 )
			# print(_mua.shape)
			# print(mua.shape)
			mua = np.concatenate(
				[mua, _mua], 2
				)

		# [medium, wavelength, ScvO2]
		# -> [ScvO2, wavelength, medium]
		mua = mua[:, :, 1:]
		mua = np.transpose(mua, (2, 1, 0))
		return mua

	def _parse_mch(self):
		
		# selected the detector_idx, pathlength, angle of the
		# photon when it is leaving the tissue

		num_media = self.header["maxmedia"]	
		# selected_list = [0] + [i for i in range(num_media+1, 2*num_media+1)] + [-1]
		selected_list = [0] + [i for i in range(2, 2 + num_media)] + [-1]
		df = self.df[:, selected_list]
		label = ["detector_idx"]
		label += ["media_{}".format(i) for i in range(self.header["maxmedia"])]
		label += ["angle"]
		df = pd.DataFrame(df, columns=label)
		return df

	@staticmethod
	def _calculate_mua(b, s, w, oxy, deoxy, water):
		mua = b * (s * oxy + (1-s) * deoxy) + w * water
		return mua

	@staticmethod
	def _load_mch(path):
		data = load_mch(path)
		# check if the mcx saved the photon seed
		if data[1]["seed_byte"] == 0:
			df, header = data
			photon_seed = None
		elif data[1]["seed_byte"] == 1:
			df, header, photon_seed = data

		return df, header, photon_seed


def test_compute_reflectance_white():
	
	df = pd.DataFrame(
		{"detector_idx":[0,0,1,1,0,1,2,3,2,3,1,1,0,2,3,3],
		 "media_0":     [1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4],
		 "media_1":     [1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
		 "angle":       [0.1,0.9,0.1,0.9,0.9,0.9,0.1,0,0.9,0,1,1,1,0,0,1]
		 }
		)
	# filter the photon that not been detected(due to critical angle)
	df = df[np.arccos(df.angle.abs()) <= 0.8]
	print(df)
	# reset the index! (cause some of them are filtered)
	df = df.reset_index(drop=True)
	print(df)
	print(df.shape)
	
	# [medium, wavelength, ScvO2]
	# -> [ScvO2, wavelength, medium]
	mua = torch.rand(100, 20, 2)
	mua = torch.tensor(mua)
	print(mua.shape)

	# [photon, medium]
	# -> [medium, photon]
	path_length = df.iloc[:, 1:-1].values
	path_length = torch.tensor(path_length).float().transpose(1, 0)
	print(path_length)
	print(path_length.shape)


	# [photon, wavelength, ScvO2]
	# -> [ScvO2, wavelength, photon]
	weight = torch.exp(-torch.matmul(mua, path_length))
	print(weight.shape)
	
	# [1, wavelength, ScvO2]
	# -> [ScvO2, wavelength, 1]
	result = torch.zeros(weight.shape[:-1]).unsqueeze(2)
	print(result.shape)

	# seperate photon with different detector
	for idx in range(4):
		# get the index of specific index
		detector_list = df.index[df["detector_idx"] == idx].tolist()

		# pick the photon that detected by specific detector
		# [1, wavelength, ScvO2]
		# -> [ScvO2, wavelength, 1]
		_weight = weight[:,:, detector_list].sum(2).unsqueeze(2)
		result = torch.cat((result, _weight), 2)
		print(result.shape)

	# [SDS, wavelength, ScvO2]
	# -> [ScvO2, wavelength, SDS]
	result = result[:, :, 1:]
	print(result.shape)

	return result


if __name__ == "__main__":
	m = MCHHandler("stuff/test.mch")

	s = time()
	result = m.compute_reflectance_white(test=True)
	e = time() - s
	print("%2d:%2d" % (e//60, e%60))
	# print(result)
	print(result.shape)
	# t = test_compute_reflectance_white()

