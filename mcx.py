import os
import sys
import json
import pickle
from glob import glob 
from random import randint
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

from utils import load_mc2
from utils.mch import MCHHandler



class MCX:
    # should be easy enough to use
    def __init__(self, config_file=None):
        if config_file is not None:
            self.load_config(config_file)
        else:
            self.config = None

    # public function

    def load_config(self, config_file):
        # load config
        with open(config_file) as f:
            self.config = json.load(f)
        with open(self.config["parameters"]) as f:
            self.parameters = json.load(f)

            # convert unit
            for key in self.parameters["geometry"].keys():
                self.parameters["geometry"][key] = self._convert_unit(self.parameters["geometry"][key])

        with open(self.config["mcx_input"]) as f:
            self.mcx_input = json.load(f)

        self.wavelength = pd.read_csv(self.config["wavelength"])["wavelength"]
        self.fiber = pd.read_csv(self.config["fiber"])

        if self.config["photon_batch"] > self.config["num_photon"]:
            self.config["photon_batch"] = self.config["num_photon"]

        # critical angle
        self.critical_angle = np.arcsin(self.config["detector_na"]/self.config["medium_n"])

        # mua
        if self.config["type"] == "ijv":
            self.mua = pd.read_csv(self.config["coefficients"])
            oxy = self.mua["oxy"].values
            deoxy = self.mua['deoxy'].values
            water = self.mua['water'].values
            collagen = self.mua['collagen'].values
            fat = self.mua['fat'].values
            melanin = self.mua['mel'].values
            wl = self.mua['wavelength'].values

            # interpolation
            oxy = np.interp(self.wavelength, wl, oxy)
            deoxy = np.interp(self.wavelength, wl, deoxy)
            collagen = np.interp(self.wavelength, wl, collagen)
            water = np.interp(self.wavelength, wl, water)
            fat = np.interp(self.wavelength, wl, fat)
            melanin = np.interp(self.wavelength, wl, melanin)

            # 1/cm --> 1/mm
            self.oxy = oxy * 0.1
            self.deoxy = deoxy * 0.1
            self.water = water * 0.1
            self.collagen = collagen * 0.1
            self.fat = fat * 0.1
            self.melanin = melanin * 0.1

        elif self.config["type"] == "phantom":
            # 校正仿體
            mua = pd.read_csv(self.config["phantom_mua"])
            musp = pd.read_csv(self.config["phantom_musp"])

            self.mua = {
                idx: np.interp(self.wavelength, mua["wl"], mua[idx]) for idx in self.config["phantom_idx"]
                }
            # since g = 0, musp = mus
            self.mus = {
                idx: np.interp(self.wavelength, musp["wl"], musp[idx]) for idx in self.config["phantom_idx"]
                }

        elif self.config["type"] == "ijv_phantom":
            # IJV仿體
            mua = pd.read_csv(self.config["phantom_mua"])
            musp = pd.read_csv(self.config["phantom_musp"])

            layers = ["skin", "fat", "ml"]
            self.mua = {
                layer: np.interp(self.wavelength, mua['wl'], mua[layer]) for layer in layers
            }

            layers = ["skin", "fat", "ml", "ijv"]
            self.mus = {
                layer: np.interp(self.wavelength, musp['wl'], musp[layer]) for layer in layers
            }

        self.session = os.path.join("output", self.config["session_id"])
        self.plot = os.path.join(self.session, 'plot')
        self.plot_mc2 = os.path.join(self.session, 'plot_mc2')
        self.result = os.path.join(self.session, 'result')
        self.mcx_output = os.path.join(self.session, 'mcx_output')
        self.json_output = os.path.join(self.session, 'json_output')

        self.reflectance = None

    def run(self, config_file=None):
        
        # load config file
        if config_file is not None:
            self.load_config(config_file)
        elif self.config is None:
            raise Exception("need to specify config file!")

        # check if the directory exist, if not make one.

        if not os.path.isdir(self.session):
            os.mkdir(self.session)

        # directory for saving plot
        if not os.path.isdir(self.plot):
            os.mkdir(self.plot)

        # directory for saving mc2 plot
        if not os.path.isdir(self.plot_mc2):
            os.mkdir(self.plot_mc2)


        # directory for saving result
        if not os.path.isdir(self.result):
            os.mkdir(self.result)

        if not os.path.isdir(self.mcx_output):
            os.mkdir(self.mcx_output)
        
        if not os.path.isdir(self.json_output):
            os.mkdir(self.json_output)

        with open(os.path.join(self.json_output, "parameters.json"), "w+") as f:
            json.dump(self.parameters, f, indent=4)
        # plot tissue
        if self.config["type"] == "ijv":
            self._plot_tissue_ijv()

        # run forward mcx here

        if self.config["type"] == "ijv":
            for idx, wl in enumerate(self.wavelength):
                self._make_input_ijv(idx)
                command = self._get_command(wl)
                print("wavelength: ", wl)
                print(command)
                sys.stdout.flush()
                os.chdir("mcx/bin")
                os.system(command)
                os.chdir("../..")

            mc2_list = glob(os.path.join(self.mcx_output, "*.mc2"))
            for mc2 in mc2_list:
                fig = plt.figure(figsize=(10,16))
                d = load_mc2(mc2, [
                    self.parameters["boundary"]["x_size"], 
                    self.parameters["boundary"]["y_size"],
                    self.parameters["boundary"]["z_size"]
                    ])
                plt.imshow(d[self.parameters["boundary"]["x_size"]//2,:,:100].T)
                name = mc2.split('/')[-1].split('.')[0]
                plt.title(name)
                plt.xlabel('y axis')
                plt.ylabel('z axis')
                plt.savefig(os.path.join(self.plot_mc2, name + ".png"))
                plt.close()


            # self.calculate_reflectance()

        elif self.config["type"] == "artery":
            self._make_input_artery(idx)

        elif self.config["type"] == "phantom":
            for idx, wl in enumerate(self.wavelength):
                for pid in self.config["phantom_idx"]:
                    self._make_input_phantom(idx, pid)
                    command = self._get_command(wl, pid)
                    print("wavelength: ", wl)
                    print("phantom: ", pid)
                    print(command)
                    os.chdir("mcx/bin")
                    os.system(command)
                    os.chdir("../..")

        elif self.config["type"] == "ijv_phantom":
            for idx, wl in enumerate(self.wavelength):
                self._make_input_ijv(idx)
                command = self._get_command(wl)
                print("wavelength: ", wl)
                print(command)
                sys.stdout.flush()
                os.chdir("mcx/bin")
                os.system(command)
                os.chdir("../..")

            # self.calculate_reflectance_phantom()

        elif self.config["type"] == "muscle":
            for idx, wl in enumerate(self.wavelength):
                # for sds_idx in range(len(self.fiber)):
                self._make_input_muscle(idx)
                command = self._get_command(wl)
                print("wavelength: ", wl)
                print(command)
                sys.stdout.flush()
                os.chdir("mcx/bin")
                os.system(command)
                os.chdir("../..")


        else:
            raise Exception("'type' in %s is invalid!\ntry 'ijv', 'artery' or 'phantom'." % config_file)

    # private function

    def _plot_tissue_ijv(self):

        skin_th = self.parameters["geometry"]["skin_thickness"]
        fat_th = self.parameters["geometry"]["fat_thickness"]
        ijv_r = self.parameters["geometry"]["ijv_radius"]
        ijv_d = self.parameters["geometry"]["ijv_depth"]
        ic_dist = self.parameters["geometry"]["ijv_cca_distance"]
        cca_r = self.parameters["geometry"]["cca_radius"]
        cca_d = self.parameters["geometry"]["cca_depth"]

        x_size = self.parameters["boundary"]["x_size"]
        y_size = self.parameters["boundary"]["y_size"]
        z_size = self.parameters["boundary"]["z_size"]

        plt.figure()
        skin = plt.Rectangle((0, 0), y_size, skin_th, fc="#FF8800", label="skin")
        fat = plt.Rectangle((0, skin_th), y_size, fat_th, fc="#BB5500", label="fat")
        muscle = plt.Rectangle((0, skin_th+fat_th), y_size, z_size-skin_th-fat_th, fc="#C63300", label="muscle")
        ijv = plt.Circle((y_size//2, ijv_d), radius=ijv_r, fc="#4169E1", label="IJV")
        cca = plt.Circle((y_size//2 - ic_dist, cca_d), radius=cca_r, fc="#800000", label="CCA")
        plt.axis([0, y_size, z_size, 0])
        plt.gca().add_patch(skin)
        plt.gca().add_patch(fat)
        plt.gca().add_patch(muscle)
        plt.gca().add_patch(ijv)
        plt.gca().add_patch(cca)
        plt.legend()
        plt.savefig(os.path.join(self.plot, self.config["session_id"] + "_geometry.png"))
        plt.close()

    def _convert_unit(self, length_mm):
        # convert mm to number of grid
        num_grid = round(length_mm/self.config["voxel_size"])
        return round(num_grid)

    def _make_input_ijv(self, wl_idx):

        mcx_input = self.mcx_input
        mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[wl_idx]
        mcx_input["Session"]["Photons"] = self.config["num_photon"]
        # optical parameter

        # 
        mcx_input["Domain"]["Media"][0]["mua"] = 0
        mcx_input["Domain"]["Media"][0]["mus"] = 0
        mcx_input["Domain"]["Media"][0]["g"] = 1
        mcx_input["Domain"]["Media"][0]["n"] = 1


        # skin
        mcx_input["Domain"]["Media"][1]["name"] = "skin"
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["skin"]["muspx"], 
            self.parameters["skin"]["bmie"],
            self.parameters["skin"]["g"]
            )
        mcx_input["Domain"]["Media"][1]["g"] = self.parameters["skin"]["g"]
        mcx_input["Domain"]["Media"][1]["n"] = self.parameters["skin"]["n"]

        # fat
        mcx_input["Domain"]["Media"][2]["name"] = "fat"
        mcx_input["Domain"]["Media"][2]["mua"] = 0
        mcx_input["Domain"]["Media"][2]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["fat"]["muspx"], 
            self.parameters["fat"]["bmie"],
            self.parameters["fat"]["g"]
            )
        mcx_input["Domain"]["Media"][2]["g"] = self.parameters["fat"]["g"]
        mcx_input["Domain"]["Media"][2]["n"] = self.parameters["fat"]["n"]

        # muscle
        mcx_input["Domain"]["Media"][3]["name"] = "muscle"
        mcx_input["Domain"]["Media"][3]["mua"] = 0
        mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["muscle"]["muspx"], 
            self.parameters["muscle"]["bmie"],
            self.parameters["muscle"]["g"]
            )
        mcx_input["Domain"]["Media"][3]["g"] = self.parameters["muscle"]["g"]
        mcx_input["Domain"]["Media"][3]["n"] = self.parameters["muscle"]["n"]
        
        # IJV
        mcx_input["Domain"]["Media"][4]["name"] = "IJV"
        mcx_input["Domain"]["Media"][4]["mua"] = 0    # for white MC
        mcx_input["Domain"]["Media"][4]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["IJV"]["muspx"], 
            self.parameters["IJV"]["bmie"],
            self.parameters["IJV"]["g"]
            )
        mcx_input["Domain"]["Media"][4]["g"] = self.parameters["IJV"]["g"]
        mcx_input["Domain"]["Media"][4]["n"] = self.parameters["IJV"]["n"]
        
        # CCA
        mcx_input["Domain"]["Media"][5]["name"] = "CCA"
        mcx_input["Domain"]["Media"][5]["mua"] = 0
        mcx_input["Domain"]["Media"][5]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["CCA"]["muspx"], 
            self.parameters["CCA"]["bmie"],
            self.parameters["CCA"]["g"]
            )
        mcx_input["Domain"]["Media"][5]["g"] = self.parameters["CCA"]["g"]
        mcx_input["Domain"]["Media"][5]["n"] = self.parameters["CCA"]["n"]


        # geometry
        skin_th = self.parameters["geometry"]["skin_thickness"]
        fat_th = self.parameters["geometry"]["fat_thickness"]
        ijv_r = self.parameters["geometry"]["ijv_radius"]
        ijv_d = self.parameters["geometry"]["ijv_depth"]
        ic_dist = self.parameters["geometry"]["ijv_cca_distance"]
        cca_r = self.parameters["geometry"]["cca_radius"]
        cca_d = self.parameters["geometry"]["cca_depth"]


        x_size = self.parameters["boundary"]["x_size"]
        y_size = self.parameters["boundary"]["y_size"]
        z_size = self.parameters["boundary"]["z_size"]

        mcx_input["Domain"]["Dim"] = [x_size, y_size, z_size]

        mcx_input["Shapes"][0]["Grid"]["Size"] = [x_size, y_size, z_size]

        # skin
        mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
        mcx_input["Shapes"][1]["Subgrid"]["Size"] = [x_size, y_size, skin_th]

        # fat
        mcx_input["Shapes"][2]["Subgrid"]["O"] = [1, 1, 1+skin_th]
        mcx_input["Shapes"][2]["Subgrid"]["Size"] = [x_size, y_size, fat_th]

        # muscle
        mcx_input["Shapes"][3]["Subgrid"]["O"] = [1, 1, 1+skin_th+fat_th]
        mcx_input["Shapes"][3]["Subgrid"]["Size"] = [x_size, y_size, z_size-skin_th-fat_th]

        # ijv 
        mcx_input["Shapes"][4]["Cylinder"]["C0"] = [x_size, y_size//2, ijv_d]
        mcx_input["Shapes"][4]["Cylinder"]["C1"] = [0, y_size//2, ijv_d]
        mcx_input["Shapes"][4]["Cylinder"]["R"] = ijv_r

        # cca 
        mcx_input["Shapes"][5]["Cylinder"]["C0"] = [x_size, y_size//2- ic_dist, cca_d]
        mcx_input["Shapes"][5]["Cylinder"]["C1"] = [0, y_size//2- ic_dist, cca_d]
        mcx_input["Shapes"][5]["Cylinder"]["R"] = cca_r


        # 改成水平！ 20190511
        src_x = 10
        mcx_input["Optode"]["Source"]["Pos"][0] = src_x
        mcx_input["Optode"]["Source"]["Pos"][1] = y_size//2

        mcx_input["Optode"]["Detector"] = []
        # 肌肉
        # for sds, r in self.fiber.values[:5]:
        #     sds = self._convert_unit(sds)
        #     r = self._convert_unit(r)
        #     det = {
        #         "R": r,
        #         "Pos": [src_x, y_size//2 + sds, 0.0]
        #     }
        #     mcx_input["Optode"]["Detector"].append(det)
        
        # IJV
        for sds, r in self.fiber.values:
            sds = self._convert_unit(sds)
            r = self._convert_unit(r)
            det = {
                "R": r,
                "Pos": [src_x + sds, y_size//2, 0.0]
            }
            mcx_input["Optode"]["Detector"].append(det)


        # set seed
        mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)

        # save the .json file in the output folder
        with open(os.path.join(self.json_output, "input_%d.json" % 
            self.wavelength[wl_idx]), 'w+') as f:
            json.dump(mcx_input, f, indent=4)

    def _make_input_ijv_phantom(self, wl_idx):
        mcx_input = self.mcx_input
        mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[wl_idx]
        mcx_input["Session"]["Photons"] = self.config["num_photon"]
        # optical parameter

        # 
        mcx_input["Domain"]["Media"][0]["mua"] = 0
        mcx_input["Domain"]["Media"][0]["mus"] = 0
        mcx_input["Domain"]["Media"][0]["g"] = 1
        mcx_input["Domain"]["Media"][0]["n"] = 1


        # skin
        mcx_input["Domain"]["Media"][1]["name"] = "skin"
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = self.mus["skin"][wl_idx]
        mcx_input["Domain"]["Media"][1]["g"] = 0
        mcx_input["Domain"]["Media"][1]["n"] = 1.4

        # fat
        mcx_input["Domain"]["Media"][2]["name"] = "fat"
        mcx_input["Domain"]["Media"][2]["mua"] = 0
        mcx_input["Domain"]["Media"][2]["mus"] = self.mus["fat"][wl_idx]
        mcx_input["Domain"]["Media"][2]["g"] = 0
        mcx_input["Domain"]["Media"][2]["n"] = 1.4

        # muscle
        mcx_input["Domain"]["Media"][3]["name"] = "muscle"
        mcx_input["Domain"]["Media"][3]["mua"] = 0
        mcx_input["Domain"]["Media"][3]["mus"] = self.mus["ml"][wl_idx]
        mcx_input["Domain"]["Media"][3]["g"] = 0
        mcx_input["Domain"]["Media"][3]["n"] = 1.4
        
        # IJV
        mcx_input["Domain"]["Media"][4]["name"] = "IJV"
        mcx_input["Domain"]["Media"][4]["mua"] = 0    # for white MC
        mcx_input["Domain"]["Media"][4]["mus"] = 0    # 墨水無散射
        mcx_input["Domain"]["Media"][4]["g"] = 0.9
        mcx_input["Domain"]["Media"][4]["n"] = 1.33
        
        # CCA
        mcx_input["Domain"]["Media"][5]["name"] = "CCA"
        mcx_input["Domain"]["Media"][5]["mua"] = 0
        mcx_input["Domain"]["Media"][5]["mus"] = self.mus["ml"][wl_idx] # 沒有CCA
        mcx_input["Domain"]["Media"][5]["g"] = 0
        mcx_input["Domain"]["Media"][5]["n"] = 1.4


        # geometry
        skin_th = self.parameters["geometry"]["skin_thickness"]
        fat_th = self.parameters["geometry"]["fat_thickness"]
        ijv_r = self.parameters["geometry"]["ijv_radius"]
        ijv_d = self.parameters["geometry"]["ijv_depth"]
        ic_dist = self.parameters["geometry"]["ijv_cca_distance"]
        cca_r = self.parameters["geometry"]["cca_radius"]
        cca_d = self.parameters["geometry"]["cca_depth"]


        x_size = self.parameters["boundary"]["x_size"]
        y_size = self.parameters["boundary"]["y_size"]
        z_size = self.parameters["boundary"]["z_size"]

        mcx_input["Domain"]["Dim"] = [x_size, y_size, z_size]

        mcx_input["Shapes"][0]["Grid"]["Size"] = [x_size, y_size, z_size]

        # skin
        mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
        mcx_input["Shapes"][1]["Subgrid"]["Size"] = [x_size, y_size, skin_th]

        # fat
        mcx_input["Shapes"][2]["Subgrid"]["O"] = [1, 1, 1+skin_th]
        mcx_input["Shapes"][2]["Subgrid"]["Size"] = [x_size, y_size, fat_th]

        # muscle
        mcx_input["Shapes"][3]["Subgrid"]["O"] = [1, 1, 1+skin_th+fat_th]
        mcx_input["Shapes"][3]["Subgrid"]["Size"] = [x_size, y_size, z_size-skin_th-fat_th]

        # ijv 
        mcx_input["Shapes"][4]["Cylinder"]["C0"] = [x_size, y_size//2, ijv_d]
        mcx_input["Shapes"][4]["Cylinder"]["C1"] = [0, y_size//2, ijv_d]
        mcx_input["Shapes"][4]["Cylinder"]["R"] = ijv_r

        # cca 
        mcx_input["Shapes"][5]["Cylinder"]["C0"] = [x_size, y_size//2- ic_dist, cca_d]
        mcx_input["Shapes"][5]["Cylinder"]["C1"] = [0, y_size//2- ic_dist, cca_d]
        mcx_input["Shapes"][5]["Cylinder"]["R"] = cca_r


        # 改成水平！ 20190511
        src_x = 10
        mcx_input["Optode"]["Source"]["Pos"][0] = src_x
        mcx_input["Optode"]["Source"]["Pos"][1] = y_size//2

        mcx_input["Optode"]["Detector"] = []
        # 肌肉
        # for sds, r in self.fiber.values[:5]:
        #     sds = self._convert_unit(sds)
        #     r = self._convert_unit(r)
        #     det = {
        #         "R": r,
        #         "Pos": [src_x, y_size//2 + sds, 0.0]
        #     }
        #     mcx_input["Optode"]["Detector"].append(det)
        
        # IJV
        for sds, r in self.fiber.values:
            sds = self._convert_unit(sds)
            r = self._convert_unit(r)
            det = {
                "R": r,
                "Pos": [src_x + sds, y_size//2, 0.0]
            }
            mcx_input["Optode"]["Detector"].append(det)


        # set seed
        mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)

        # save the .json file in the output folder
        with open(os.path.join(self.json_output, "input_%d.json" % 
            self.wavelength[wl_idx]), 'w+') as f:
            json.dump(mcx_input, f, indent=4)


    def _make_input_muscle(self, wl_idx):

        mcx_input = self.mcx_input
        mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[wl_idx]

        # optical parameter

        # 
        mcx_input["Domain"]["Media"][0]["mua"] = 0
        mcx_input["Domain"]["Media"][0]["mus"] = 0
        mcx_input["Domain"]["Media"][0]["g"] = 1
        mcx_input["Domain"]["Media"][0]["n"] = 1

        # skin
        mcx_input["Domain"]["Media"][1]["name"] = "skin"
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["skin"]["muspx"], 
            self.parameters["skin"]["bmie"],
            self.parameters["skin"]["g"]
            )
        mcx_input["Domain"]["Media"][1]["g"] = self.parameters["skin"]["g"]
        mcx_input["Domain"]["Media"][1]["n"] = self.parameters["skin"]["n"]

        # fat
        mcx_input["Domain"]["Media"][2]["name"] = "fat"
        mcx_input["Domain"]["Media"][2]["mua"] = 0
        mcx_input["Domain"]["Media"][2]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["fat"]["muspx"], 
            self.parameters["fat"]["bmie"],
            self.parameters["fat"]["g"]
            )
        mcx_input["Domain"]["Media"][2]["g"] = self.parameters["fat"]["g"]
        mcx_input["Domain"]["Media"][2]["n"] = self.parameters["fat"]["n"]

        # muscle
        mcx_input["Domain"]["Media"][3]["name"] = "muscle"
        mcx_input["Domain"]["Media"][3]["mua"] = 0
        mcx_input["Domain"]["Media"][3]["mus"] = self._calculate_mus(
            wl_idx,
            self.parameters["muscle"]["muspx"], 
            self.parameters["muscle"]["bmie"],
            self.parameters["muscle"]["g"]
            )
        mcx_input["Domain"]["Media"][3]["g"] = self.parameters["muscle"]["g"]
        mcx_input["Domain"]["Media"][3]["n"] = self.parameters["muscle"]["n"]
        

        # geometry
        skin_th = self.parameters["geometry"]["skin_thickness"]
        fat_th = self.parameters["geometry"]["fat_thickness"]


        x_size = self.parameters["boundary"]["x_size"]
        y_size = self.parameters["boundary"]["y_size"]
        z_size = self.parameters["boundary"]["z_size"]

        mcx_input["Domain"]["Dim"] = [x_size, y_size, z_size]

        mcx_input["Shapes"][0]["Grid"]["Size"] = [x_size, y_size, z_size]

        # prism
        prism_th = 20
        mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
        mcx_input["Shapes"][1]["Subgrid"]["Size"] = [x_size, y_size, prism_th]


        # skin
        mcx_input["Shapes"][2]["Subgrid"]["O"] = [1, 1, 1+prism_th]
        mcx_input["Shapes"][2]["Subgrid"]["Size"] = [x_size, y_size, skin_th]

        # fat
        mcx_input["Shapes"][3]["Subgrid"]["O"] = [1, 1, 1+skin_th+prism_th]
        mcx_input["Shapes"][3]["Subgrid"]["Size"] = [x_size, y_size, fat_th]

        # muscle
        mcx_input["Shapes"][4]["Subgrid"]["O"] = [1, 1, 1+skin_th+fat_th+prism_th]
        mcx_input["Shapes"][4]["Subgrid"]["Size"] = [x_size, y_size, z_size-skin_th-fat_th]

        # load fiber
        # 改成水平！ 20190517
        src_x = 10
        mcx_input["Optode"]["Source"]["Pos"][0] = src_x
        mcx_input["Optode"]["Source"]["Pos"][1] = y_size//2

        mcx_input["Optode"]["Detector"] = []

        
        for sds, r in self.fiber.values:
            sds = self._convert_unit(sds)
            r = self._convert_unit(r)
            det = {
                "R": r,
                "Pos": [src_x + sds, y_size//2, 0.0]
            }
            mcx_input["Optode"]["Detector"].append(det)


        # set seed
        mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)

        # save the .json file in the output folder
        with open(os.path.join(self.json_output, "input_%d.json" % (self.wavelength[wl_idx])), 'w+') as f:
            json.dump(mcx_input, f, indent=4)

    def _make_input_artery(self, wl_idx):
        
        raise NotImplementedError

    def _make_input_phantom(self, wl_idx, phantom_idx):
        mcx_input = self.mcx_input

        mcx_input["Session"]["ID"] = self.config["session_id"] + "_%d" % self.wavelength[wl_idx]
        mcx_input["Session"]["Photons"] = self.config["num_photon"]

        # optical parameter
 
        mcx_input["Domain"]["Media"][0]["mua"] = 0
        mcx_input["Domain"]["Media"][0]["mus"] = 0
        mcx_input["Domain"]["Media"][0]["g"] = 1
        mcx_input["Domain"]["Media"][0]["n"] = 1


        # phantom
        mcx_input["Domain"]["Media"][1]["name"] = "phantom"
        # mcx_input["Domain"]["Media"][1]["mua"] = self.mua[phantom_idx][wl_idx]
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = self.mus[phantom_idx][wl_idx]
        mcx_input["Domain"]["Media"][1]["g"] = self.parameters["phantom"]["g"]
        mcx_input["Domain"]["Media"][1]["n"] = self.parameters["phantom"]["n"]



        x_size = self.parameters["boundary"]["x_size"]
        y_size = self.parameters["boundary"]["y_size"]
        z_size = self.parameters["boundary"]["z_size"]

        mcx_input["Domain"]["Dim"] = [x_size, y_size, z_size]


        mcx_input["Shapes"][0]["Grid"]["Size"] = [x_size, y_size, z_size]


        # phantom
        mcx_input["Shapes"][1]["Subgrid"]["O"] = [1, 1, 1]
        mcx_input["Shapes"][1]["Subgrid"]["Size"] = [x_size, y_size, 100]


        # load fiber
        src_x = 10
        mcx_input["Optode"]["Source"]["Pos"][0] = src_x
        mcx_input["Optode"]["Source"]["Pos"][1] = y_size//2

        mcx_input["Optode"]["Detector"] = []
        for sds, r in self.fiber.values:
            sds = self._convert_unit(sds)
            r = self._convert_unit(r)
            det = {
                "R": r,
                "Pos": [src_x + sds, y_size//2, 0.0]
            }
            mcx_input["Optode"]["Detector"].append(det)


        # set seed
        mcx_input["Session"]["RNGSeed"] = randint(0, 1000000000)


        # save the .json file in the output folder
        with open(os.path.join(self.json_output, "input_{}_{}.json".format(
            self.wavelength[wl_idx], phantom_idx
            )), 'w+') as f:
            json.dump(mcx_input, f, indent=4)

    def _get_command(self, wl_idx, idx=None):
        # create the command for mcx
        if idx:
            session_name = "\"%s_%d_%s\" " % (self.config["session_id"], wl_idx, str(idx))
            geometry_file = "\"%s\" " % os.path.abspath(
                os.path.join(self.json_output, "input_{}_{}.json".format(wl_idx, idx))
                )
        else:
            session_name = "\"%s_%d\" " % (self.config["session_id"], wl_idx)
            geometry_file = "\"%s\" " % os.path.abspath(
                os.path.join(self.json_output, "input_%d.json" % (wl_idx))
                )
        root = "\"%s\" " % os.path.join(os.path.abspath(self.session), "mcx_output")
        unitmm = "%f " % self.config["voxel_size"]
        photon = "%d " % self.config["photon_batch"]
        num_batch = "%d " % (self.config["num_photon"]//self.config["photon_batch"])
        maxdetphoton = "10000000"
        # maxdetphoton = "%d" % (self.config["num_photon"]//5)
        # save_mc2 = "0 " if self.config["train"] else "1 "
        # mc2 is seldom used
        save_mc2 = "0 "

        if os.name == "posix":
            # linux
            command = "./mcx"
        elif os.name == "nt":
            # windows
            command = "mcx.exe"
        else:
            command = "./mcx"
        command += " --session " + session_name +\
        "--input " + geometry_file +\
        "--root " + root +\
        "--gpu 1 " +\
        "--autopilot 1 " +\
        "--photon " + photon +\
        "--repeat " + num_batch +\
        "--normalize 1 " +\
        "--save2pt " + save_mc2 +\
        "--reflect 1 " +\
        "--savedet 1 " +\
        "--saveexit 1 " +\
        "--unitinmm " + unitmm +\
        "--saveseed 0 " +\
        "--skipradius -2 " +\
        "--array 0 " +\
        "--dumpmask 0 " +\
        "--maxdetphoton " + maxdetphoton
        
        return command

    def _calculate_mua(self, idx, b, s, w, f, m):
        mua = b * (s * self.oxy[idx] + (1-s) * self.deoxy[idx]) + w * self.water[idx] + f * self.fat[idx] + m * self.melanin[idx]
        # print("==============================")
        # print("wl: ", self.wavelength[idx])
        # print("oxy: ", self.oxy[idx])
        # print("deoxy: ", self.deoxy[idx])
        # print("water: ", self.water[idx])
        # print("mua: ", mua)
        return mua

    def _calculate_muscle_mua(self, idx, w):
        mua = w * self.water[idx] + (1-w) * self.collagen[idx]
        return mua

    def _calculate_mus(self, idx, mus500, bmie, g):
        wl = self.wavelength[idx]
        # mus_p = mus500 * (fray * (wl/500)**(-4) + (1-fray) * (wl/500) ** (-bmie))
        mus_p = mus500 * (wl/500) ** (-bmie)

        mus = mus_p/(1-g) * 0.1
        return mus 

    def _calculate_sens(self, data):
        # data -> [ScvO2, SDS, wl]
        def cal_sen(x1, x2):
            return np.abs(x2 - x1).mean()/(x1).mean()

        def cal_sen2(x1, x2):
            return np.abs(x2 - x1)/x1

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
        plt.ylabel("sensitivity")
        plt.xlabel("ScvO2")

        plt.savefig(os.path.join(self.plot, "sens_scvo2.png"))
        plt.clf()

        # 2
        plt.figure(figsize=(12,8))
        sens = np.asarray(sensitivity)

        plt.imshow(sens, aspect='auto')
        plt.title("")
        plt.xlabel("ScvO2")
        plt.ylabel("SDS")
        plt.colorbar()
        plt.savefig(os.path.join(self.plot, "sens_heatmap.png"))
        plt.clf()

        # 3
        for s in range(num_sds):
            sww = []
            for p in range(100):
                sw = []
                for w in range(36):
                    sw.append(cal_sen2(data[i, s, w], data[i+1, s, w]))
                sww.append(sw)
            sww = np.asarray(sww)
            plt.figure(figsize=(18, 10))
            plt.imshow(sww, aspect='auto')
            plt.xticks([i for i in range(36)], labels=wl)
            plt.colorbar()
            plt.xlabel("wavelength")
            plt.ylabel("ScvO2")
            plt.title("SDS #%d" % s)
            plt.savefig(os.path.join(self.plot, "wl_hm_%d.png" % s))
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
        plt.savefig(os.path.join(self.plot, "sens_wl.png"))


if __name__ == "__main__":
    mcx = MCX()
