import os 
import json 
import random 
import string 

import numpy as np 
import pandas as pd 
from MySQLdb import connect

from mcx import MCX 
from utils.mch import MCHHandler


class MCXGen(MCX):
    def __init__(self):
        self.load_config("configs/train.json")
        self.mch = MCHHandler()
        self.absorb_num = 100


    def train(self):
        # 1/mm
        optics_range = {
            "skin_mus": (0, 5),
            "skin_mua": (0.001, 0.1),
            "skin_g": (0.6, 0.9),
            "skin_n": (1.38, 1.45),

            "fat_mus": (0.1, 5),
            "fat_mua": (0.001, 0.05),
            "fat_g": (0.8, 0.99),
            "fat_n": (1.38, 1.45),

            "muscle_mus": (0.1, 5),
            "muscle_mua": (0.01, 0.1),
            "muscle_g": (0.8, 0.99),
            "muscle_n": (1.38, 1.45),

            "ijv_mus": (0.1, 5),
            "ijv_mua": (0.01, 0.5),
            "ijv_g": (0.8, 0.99),
            "ijv_n": (1.38, 1.45),

            "cca_mus": (0.1, 5),
            "cca_mua": (0.01, 0.1),
            "cca_g": (0.8, 0.99),
            "cca_n": (1.38, 1.45),
        }

        geo_range = {
        # mm
            "geo_skin": (0.5, 1.5),
            "geo_fat": (0.5, 1.5),
            "geo_ijvr": (2.5, 10.5),
            "geo_ijvd": (2, 16),
            "geo_ccar": (2.5, 4.5),
            "geo_ccad": (5, 26),
            "geo_ijvcca": (5, 10)
        }


        # mus
        skin_mus = random.uniform(optics_range["skin_mus"][0], optics_range["skin_mus"][1])
        skin_g = random.uniform(optics_range["skin_g"][0], optics_range["skin_g"][1])
        skin_n = random.uniform(optics_range["skin_n"][0], optics_range["skin_n"][1])
        
        fat_mus = random.uniform(optics_range["fat_musp"][0], optics_range["fat_musp"][1])
        fat_g = random.uniform(optics_range["fat_g"][0], optics_range["fat_g"][1])
        fat_n = random.uniform(optics_range["fat_n"][0], optics_range["fat_n"][1])
        
        muscle_mus = random.uniform(optics_range["muscle_musp"][0], optics_range["muscle_musp"][1])
        muscle_g = random.uniform(optics_range["muscle_g"][0], optics_range["muscle_g"][1])
        muscle_n = random.uniform(optics_range["muscle_n"][0], optics_range["muscle_n"][1])
        
        ijv_mus = random.uniform(optics_range["ijv_musp"][0], optics_range["ijv_musp"][1])
        ijv_g = random.uniform(optics_range["ijv_g"][0], optics_range["ijv_g"][1])
        ijv_n = random.uniform(optics_range["ijv_n"][0], optics_range["ijv_n"][1])
        
        cca_mus = random.uniform(optics_range["cca_musp"][0], optics_range["cca_musp"][1])
        cca_g = random.uniform(optics_range["cca_g"][0], optics_range["cca_g"][1])
        cca_n = random.uniform(optics_range["cca_n"][0], optics_range["cca_n"][1])
                
        # geo

        geo_skin = random.uniform(geo_range["geo_skin"][0], geo_range["geo_skin"][1])
        geo_fat = random.uniform(geo_range["geo_fat"][0], geo_range["geo_fat"][1])
        geo_ijvd = random.uniform(geo_range["geo_ijvd"][0], geo_range["geo_ijvd"][1])
        geo_ijvr = random.uniform(geo_range["geo_ijvr"][0], geo_range["geo_ijvr"][1])
        while geo_skin + geo_fat > geo_ijvd - geo_ijvr:
            geo_ijvd = random.uniform(geo_range["geo_ijvd"][0], geo_range["geo_ijvd"][1])
            geo_ijvr = random.uniform(geo_range["geo_ijvr"][0], geo_range["geo_ijvr"][1])

        geo_ccad = random.uniform(geo_range["geo_ccad"][0], geo_range["geo_ccad"][1])
        geo_ccar = random.uniform(geo_range["geo_ccar"][0], geo_range["geo_ccar"][1])
        geo_ijvcca = random.uniform(geo_range["geo_ijvcca"][0], geo_range["geo_ijvcca"][1])
        while (geo_ccad - geo_ijvd)**2 + (geo_ijvcca)**2 <  (geo_ccar + geo_ijvr)**2:
            geo_ccad = random.uniform(geo_range["geo_ccad"][0], geo_range["geo_ccad"][1])
            geo_ccar = random.uniform(geo_range["geo_ccar"][0], geo_range["geo_ccar"][1])
            geo_ijvcca = random.uniform(geo_range["geo_ijvcca"][0], geo_range["geo_ijvcca"][1])


        with open(self.config["parameters"]) as f:
            inp = json.load(f)

        inp["skin"]["muspx"] = skin_mus
        inp["skin"]["mua"] = 0
        inp["skin"]["g"] = skin_g
        inp["skin"]["n"] = skin_n

        inp["fat"]["muspx"] = fat_musp
        inp["fat"]["mua"] = 0
        inp["fat"]["g"] = fat_g
        inp["fat"]["n"] = fat_n

        inp["muscle"]["muspx"] = muscle_musp
        inp["muscle"]["mua"] = 0
        inp["muscle"]["g"] = muscle_g
        inp["muscle"]["n"] = muscle_n

        inp["IJV"]["muspx"] = ijv_musp
        inp["IJV"]["mua"] = 0
        inp["IJV"]["g"] = ijv_g
        inp["IJV"]["n"] = ijv_n

        inp["CCA"]["muspx"] = cca_musp
        inp["CCA"]["mua"] = 0
        inp["CCA"]["g"] = cca_g
        inp["CCA"]["n"] = cca_n

        inp["geometry"]["skin_thickness"] = self._convert_unit(geo_skin)
        inp["geometry"]["fat_thickness"] = self._convert_unit(geo_fat)
        inp["geometry"]["ijv_radius"] = self._convert_unit(geo_ijvr)
        inp["geometry"]["ijv_depth"] = self._convert_unit(geo_ijvd)
        inp["geometry"]["cca_radius"] = self._convert_unit(geo_ccar)
        inp["geometry"]["cca_depth"] = self._convert_unit(geo_ccad)
        inp["geometry"]["ijv_cca_distance"] = self._convert_unit(geo_ijvcca)
        inp["boundary"]["x_size"] = 150
        inp["boundary"]["y_size"] = 150
        inp["boundary"]["z_size"] = 200


        # make input
        idx = self._get_idx()

        mcx_input = self.mcx_input
        mcx_input["Session"]["ID"] = idx
        mcx_input["Session"]["Photons"] = self.config["num_photon"]


        # 
        mcx_input["Domain"]["Media"][0]["mua"] = 0
        mcx_input["Domain"]["Media"][0]["mus"] = 0
        mcx_input["Domain"]["Media"][0]["g"] = 1
        mcx_input["Domain"]["Media"][0]["n"] = 1


        # skin
        mcx_input["Domain"]["Media"][1]["name"] = "skin"
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = skin_mus
        mcx_input["Domain"]["Media"][1]["g"] = inp["skin"]["g"]
        mcx_input["Domain"]["Media"][1]["n"] = inp["skin"]["n"]

        # fat
        mcx_input["Domain"]["Media"][2]["name"] = "fat"
        mcx_input["Domain"]["Media"][2]["mua"] = 0
        mcx_input["Domain"]["Media"][2]["mus"] = fat_mus
        mcx_input["Domain"]["Media"][2]["g"] = inp["fat"]["g"]
        mcx_input["Domain"]["Media"][2]["n"] = inp["fat"]["n"]

        # muscle
        mcx_input["Domain"]["Media"][3]["name"] = "muscle"
        mcx_input["Domain"]["Media"][3]["mua"] = 0
        mcx_input["Domain"]["Media"][3]["mus"] = muscle_mus
        mcx_input["Domain"]["Media"][3]["g"] = inp["muscle"]["g"]
        mcx_input["Domain"]["Media"][3]["n"] = inp["muscle"]["n"]
        
        # IJV
        mcx_input["Domain"]["Media"][4]["name"] = "IJV"
        mcx_input["Domain"]["Media"][4]["mua"] = 0
        mcx_input["Domain"]["Media"][4]["mus"] = ijv_mus 
        mcx_input["Domain"]["Media"][4]["g"] = inp["IJV"]["g"]
        mcx_input["Domain"]["Media"][4]["n"] = inp["IJV"]["n"]
        
        # CCA
        mcx_input["Domain"]["Media"][5]["name"] = "CCA"
        mcx_input["Domain"]["Media"][5]["mua"] = 0
        mcx_input["Domain"]["Media"][5]["mus"] = cca_mus 
        mcx_input["Domain"]["Media"][5]["g"] = inp["CCA"]["g"]
        mcx_input["Domain"]["Media"][5]["n"] = inp["CCA"]["n"]


        # geometry
        skin_th = inp["geometry"]["skin_thickness"]
        fat_th = inp["geometry"]["fat_thickness"]
        ijv_r = inp["geometry"]["ijv_radius"]
        ijv_d = inp["geometry"]["ijv_depth"]
        ic_dist = inp["geometry"]["ijv_cca_distance"]
        cca_r = inp["geometry"]["cca_radius"]
        cca_d = inp["geometry"]["cca_depth"]


        x_size = inp["boundary"]["x_size"]
        y_size = inp["boundary"]["y_size"]
        z_size = inp["boundary"]["z_size"]

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
        mcx_input["Session"]["RNGSeed"] = random.randint(0, 1000000000)

        # save the .json file in the output folder
        with open("train/input_mcx_{}.json".format(idx), 'w+') as f:
            json.dump(mcx_input, f, indent=4)

        command = self._get_command(idx)
        print(command)
        os.chdir("mcx/bin")
        os.system(command)
        os.chdir("../..")

        # run WMC
        mua = []
        for i in range(self.absorb_num):
            skin_mua = random.uniform(optics_range["skin_mua"][0], optics_range["skin_mua"][1])

            fat_mua = random.uniform(optics_range["fat_mua"][0], optics_range["fat_mua"][1])

            muscle_mua = random.uniform(optics_range["muscle_mua"][0], optics_range["muscle_mua"][1])

            ijv_mua = random.uniform(optics_range["ijv_mua"][0], optics_range["ijv_mua"][1])

            cca_mua = random.uniform(optics_range["cca_mua"][0], optics_range["cca_mua"][1])

            mua += [np.array([skin_mua, fat_mua, muscle_mua, ijv_mua, cca_mua])]

        mua = np.array(mua).T

        spec = self.mch.run_wmc_train(os.path.join("train", "mch", "{}.mch".format(idx)), args)



        # save result to DB
        for i in range(self.absorb_num):


            field = "idx, "
            field += "skin_mua, skin_mus, skin_g, skin_n, "
            field += "fat_mua, fat_mus, fat_g, fat_n, "
            field += "muscle_mua, muscle_mus, muscle_g, muscle_n, "
            field += "ijv_mua, ijv_mus, ijv_g, ijv_n, "
            field += "cca_mua, cca_mus, cca_g, cca_n, "
            field += "skin_thickness, fat_thickness, ijv_radius, ijv_depth, cca_radius, cca_depth, ijv_cca_distance, "
            field += "reflectance_20, reflectance_24, reflectance_28"

            values = "'{}', ".format(idx)
            values += "'{}', '{}', '{}', '{}', ".format(mua[i][0], skin_mus, skin_g, skin_n)
            values += "'{}', '{}', '{}', '{}', ".format(mua[i][1], fat_mus, fat_g, fat_n)
            values += "'{}', '{}', '{}', '{}', ".format(mua[i][2], muscle_mus, muscle_g, muscle_n)
            values += "'{}', '{}', '{}', '{}', ".format(mua[i][3], ijv_mus, ijv_g, ijv_n)
            values += "'{}', '{}', '{}', '{}', ".format(mua[i][4], cca_mus, cca_g, cca_n)
            values += "'{}', '{}', '{}', '{}', '{}', '{}', '{}', ".format(
                geo_skin, geo_fat, geo_ijvr, geo_ijvd, geo_ccar, geo_ccad, geo_ijvcca
            )
            values += "'{}', '{}', '{}'".format(spec[i][0], spec[i][1], spec[i][2])


            conn = connect(
                host="140.112.174.26",
                user="md703",
                passwd=os.getenv("PASSWD"),
                db="ijv"
            )

            sql = "INSERT INTO ijv_ann_3({}) VALUES({})".format(field, values)
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()


    def _get_command(self, idx):
        # create the command for mcx

        session_name = "\"{}\" ".format(idx)
        geometry_file = "\"{}\" ".format(os.path.abspath(os.path.join("train", "input_mcx_{}.json".format(idx))))
        root = "\"{}\" ".format(os.path.abspath(os.path.join("train", "mch")))
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


    @staticmethod
    def _get_idx(n=10):
        assert n > 1, "n should be greater than 1"
        first = random.choice(string.ascii_letters)
        rest = "".join(random.sample(string.ascii_letters + string.digits, n-1))
        return first + rest



if __name__ == "__main__":
    import sys
    mcx = MCXGen()
    for i in range(int(sys.argv[1])):
        mcx.train()