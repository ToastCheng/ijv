import os
import json
import pickle
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from mchhandler_test import MCHHandler


def get_now_time():
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	return now


class MCX:
	# a class that use as an interface
	def __init__(self, config_file="config.json"):
		with open(config_file) as f:
			self.config = json.load(f)
		with open(self.config["input_file"]) as f:
			self.parameters = json.load(f)

		self.wavelength = pd.read_csv(self.config["wavelength_file"])["wavelength"]
		self.mua = pd.read_csv(self.config["absorb_file"])

		self.handler = MCHHandler()

		# handle the situation that batch > total
		if self.config["photon_batch"] > self.config["num_photon"]:
			self.config["photon_batch"] = self.config["num_photon"] 

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
		self.oxy = oxy * 0.1
		self.deoxy = deoxy * 0.1
		self.water = water *  0.1

		self.session = self.session = os.path.join(self.config["save_path"], self.config["session_id"])
		self.plot = os.path.join(self.session, 'plot')
		self.result = os.path.join(self.session, 'result')
		self.mcx_output = os.path.join(self.session, 'mcx_output')

	def run(self):
		# run the MCX simulation

		# save_path (root)
		if not os.path.isdir(self.config["save_path"]):
			os.mkdir(self.config["save_path"])
		
		# session name
		if not os.path.isdir(self.session):
			os.mkdir(self.session)

		# directory for saving plot
		if not os.path.isdir(self.plot):
			os.mkdir(self.plot)

		# directory for saving result
		if not os.path.isdir(self.result):
			os.mkdir(self.result)

		if not os.path.isdir(self.mcx_output):
			os.mkdir(self.mcx_output)


		for idx, wl in tqdm(enumerate(self.wavelength)):
			command = self._get_command(wl)
			self._make_mcx_input(idx)
			os.chdir("mcx/bin")
			os.system(command)
			os.chdir("../..")

	def _make_mcx_input(self, idx):
		with open(self.config["geometry_file"]) as f:
			# load a template first
			mcx_input = json.load(f)

		mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[idx]


		mcx_input["Domain"]["Media"][0]["mua"] = 0
		mcx_input["Domain"]["Media"][0]["mus"] = 0
		mcx_input["Domain"]["Media"][0]["g"] = 1
		mcx_input["Domain"]["Media"][0]["n"] = 1

		mcx_input["Domain"]["Media"][1]["name"] = "muscle"
		mcx_input["Domain"]["Media"][1]["mua"] = self._calculate_mua(
			idx, 
			self.parameters["muscle"]["blood_volume_fraction"], 
			self.parameters["muscle"]["ScvO2"], 
			self.parameters["muscle"]["water_volume"]
			)
		mcx_input["Domain"]["Media"][1]["mus"] = self._calculate_mus(
			idx,
			self.parameters["muscle"]["muspx"], 
			self.parameters["muscle"]["fray"], 
			self.parameters["muscle"]["bmie"],
			self.parameters["muscle"]["g"]
			)
		mcx_input["Domain"]["Media"][1]["g"] = self.parameters["muscle"]["g"]
		mcx_input["Domain"]["Media"][1]["n"] = self.parameters["muscle"]["n"]
		

		mcx_input["Domain"]["Media"][2]["name"] = "IJV"
		mcx_input["Domain"]["Media"][2]["mua"] = 0	# for white MC
		mcx_input["Domain"]["Media"][2]["mus"] = self._calculate_mus(
			idx,
			self.parameters["IJV"]["muspx"], 
			self.parameters["IJV"]["fray"], 
			self.parameters["IJV"]["bmie"],
			self.parameters["IJV"]["g"]
			)
		mcx_input["Domain"]["Media"][2]["g"] = self.parameters["IJV"]["g"]
		mcx_input["Domain"]["Media"][2]["n"] = self.parameters["IJV"]["n"]
		

		mcx_input["Domain"]["Media"][3]["name"] = "CCA"
		mcx_input["Domain"]["Media"][3]["mua"] = self._calculate_mua(
			idx,
			self.parameters["CCA"]["blood_volume_fraction"], 
			self.parameters["CCA"]["ScvO2"], 
			self.parameters["CCA"]["water_volume"]
			)
		mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
			idx,
			self.parameters["CCA"]["muspx"], 
			self.parameters["CCA"]["fray"], 
			self.parameters["CCA"]["bmie"],
			self.parameters["CCA"]["g"]
			)
		mcx_input["Domain"]["Media"][3]["g"] = self.parameters["CCA"]["g"]
		mcx_input["Domain"]["Media"][3]["n"] = self.parameters["CCA"]["n"]

		# save the .json file in the output folder
		with open(self.config["geometry_file"], 'w+') as f:
			json.dump(mcx_input, f, indent=4)

	def _calculate_mua(self, idx, b, s, w):
		mua = b * (s * self.oxy[idx] + (1-s) * self.deoxy[idx]) + w * self.water[idx]
		print("==============================")
		print("wl: ", self.wavelength[idx])
		print("oxy: ", self.oxy[idx])
		print("deoxy: ", self.deoxy[idx])
		print("water: ", self.water[idx])
		print("mua: ", mua)
		return mua

	def _calculate_mus(self, idx, mus500, fray, bmie, g):
		wl = self.wavelength[idx]
		mus_p = mus500 * (fray * (wl/500)**(-4) + (1-fray) * (wl/500) ** (-bmie))
		mus = mus_p/g * 0.1
		return mus 

	def _get_command(self, wl):
		# dummy function create the command for mcx
		session_name = "\"%s_%d\" " % (self.config["session_id"], wl)
		geometry_file = "\"%s\" " % os.path.abspath(self.config["geometry_file"])
		root = "\"%s\" " % os.path.join(os.path.abspath(self.session), "mcx_output")
		unitmm = "%f " % self.config["voxel_size"]
		photon = "%d " % self.config["photon_batch"]
		num_batch = "%d " % (self.config["num_photon"]//self.config["photon_batch"])
		maxdetphoton = "%d" % (self.config["num_photon"]//2)

		command = \
		"./mcx --session " + session_name +\
		"--input " + geometry_file +\
		"--root " + root +\
		"--gpu 1 " +\
		"--autopilot 1 " +\
		"--photon " + photon +\
		"--repeat " + num_batch +\
		"--normalize 1 " +\
		"--save2pt 1 " +\
		"--reflect 1 " +\
		"--savedet 1 " +\
		"--saveexit 1 " +\
		"--unitinmm " + unitmm +\
		"--saveseed 0 " +\
		"--seed '1648335518' " +\
		"--skipradius -2 " +\
		"--array 0 " +\
		"--dumpmask 0 " +\
		"--maxdetphoton " + maxdetphoton
		print(command)
		return command

	def calculate_reflectance(self, plot=True, verbose=True, save=True):
		# This function should called after 
		# mcx.run(), or the .mch files are outputed by mcx
		results = []
		for wl in self.wavelength:
			result = self.handler.compute_reflectance_white(
				wl=wl,
				mch_file=os.path.join(self.mcx_output, "%s_%d.mch" % (self.config["session_id"], wl))
				)
			results.append(result)
		
		# [wl, SDS, ScvO2]
		# -> [ScvO2, SDS, wl]
		results = np.asarray(results).transpose(2, 1, 0)
		# print(results.shape)

		if plot:
			for r_idx, r in enumerate(results):
				fig = plt.figure()
				for d_idx, d in enumerate(r):
					plt.plot(self.wavelength, np.log(d), label="detector %d" % d_idx)
				
				path = os.path.join(self.plot, "Scv_%d.png" % r_idx)
				plt.title('Scv_%d' % r_idx)
				plt.legend()
				plt.savefig(path)
				plt.close()

		if save:
			path = os.path.join(self.result, "result.pkl")
			with open(path, 'wb') as f:
				pickle.dump(results, f)

	def reload_result(self, path):
		with open(path, 'rb') as f:
			results = pickle.load(f)
		return results


def test_medium():
	mcx = MCX()
	command = mcx._get_command(400)
	os.chdir("mcx/bin")
	os.system(command)


def test_pickle():
	mcx = MCX()
	path = "output/test/result/" 
	results = mcx.reload_result(path+"result.pkl")
	
	for r_idx, r in enumerate(results):
		fig = plt.figure()
		for d_idx, d in enumerate(r):
			plt.plot(mcx.wavelength, np.log10(d), label="detector %d" % d_idx)
		
		_path = os.path.join(path, "Scv_%d.png" % r_idx)
		plt.title('Scv_%d' % r_idx)
		plt.legend()
		plt.savefig(_path)
		plt.close()


if __name__ == "__main__":
	# test_medium()
	test_pickle()