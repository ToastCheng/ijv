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
        self.input = {
            "skin_mua": None,
            "skin_mus": None,
            "skin_g": None,
            "skin_n": None,

            "fat_mua": None,
            "fat_mus": None,
            "fat_g": None,
            "fat_n": None,

            "muscle_mua": None,
            "muscle_mus": None,
            "muscle_g": None,
            "muscle_n": None,

            "ijv_mua": None,
            "ijv_mus": None,
            "ijv_g": None,
            "ijv_n": None,

            "cca_mua": None,
            "cca_mus": None,
            "cca_g": None,
            "cca_n": None,

            "skin_thickness": None,
            "fat_thickness": None,
            "ijv_radius": None,
            "ijv_depth": None,
            "cca_radius": None,
            "cca_depth": None,
            "ijv_cca_distance": None
        }
        self.mch = MCHHandler()


    def train(self):
        # 1/cm
        mus_range = {
            "skin_musp": (0, 30),
            "skin_bmie": (0.01, 2.453),
            "skin_g": (0.6, 0.9),
            "skin_n": (1.38, 1.45),

            "fat_musp": (1.7, 35.8),
            "fat_bmie": (0.085, 0.988),
            "fat_g": (0.8, 0.99),
            "fat_n": (1.38, 1.45),

            "muscle_musp": (9.8, 40.0),
            "muscle_bmie": (0.01, 2.82),
            "muscle_g": (0.8, 0.99),
            "muscle_n": (1.38, 1.45),

            "ijv_musp": (1, 20),
            "ijv_bmie": (0.5, 1),
            "ijv_g": (0.8, 0.99),
            "ijv_n": (1.38, 1.45),

            "cca_musp": (1, 20),
            "cca_bmie": (0.5, 1),
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

        mua_range = {
            "skin_b": (0.0012, 0.8),
            "skin_s": (0.7, 1.0),
            "skin_w": (0.01, 0.8),
            "skin_f": (0.01, 0.8),
            "skin_m": (0.0087, 0.8),

            "fat_b": (0.0, 0.1),
            "fat_s": (0.5, 0.9),
            "fat_f": (0.8, 1)

            "muscle_b": (0.005, 0.5),
            "muscle_s": (0.5, 0.9),
            "muscle_w": (0.5, 0.9),

            "ijv_s": (0.0, 1.0),

            "cca_s": (0.0, 1.0),
        }

        # mus
        skin_musp = random.uniform(mus_range["skin_musp"][0], mus_range["skin_musp"][1])
        skin_bmie = random.uniform(mus_range["skin_bmie"][0], mus_range["skin_bmie"][1])
        skin_g = random.uniform(mus_range["skin_g"][0], mus_range["skin_g"][1])
        skin_n = random.uniform(mus_range["skin_n"][0], mus_range["skin_n"][1])
        
        fat_musp = random.uniform(mus_range["fat_musp"][0], mus_range["fat_musp"][1])
        fat_bmie = random.uniform(mus_range["fat_bmie"][0], mus_range["fat_bmie"][1])
        fat_g = random.uniform(mus_range["fat_g"][0], mus_range["fat_g"][1])
        fat_n = random.uniform(mus_range["fat_n"][0], mus_range["fat_n"][1])
        
        muscle_musp = random.uniform(mus_range["muscle_musp"][0], mus_range["muscle_musp"][1])
        muscle_bmie = random.uniform(mus_range["muscle_bmie"][0], mus_range["muscle_bmie"][1])
        muscle_g = random.uniform(mus_range["muscle_g"][0], mus_range["muscle_g"][1])
        muscle_n = random.uniform(mus_range["muscle_n"][0], mus_range["muscle_n"][1])
        
        ijv_musp = random.uniform(mus_range["ijv_musp"][0], mus_range["ijv_musp"][1])
        ijv_bmie = random.uniform(mus_range["ijv_bmie"][0], mus_range["ijv_bmie"][1])
        ijv_g = random.uniform(mus_range["ijv_g"][0], mus_range["ijv_g"][1])
        ijv_n = random.uniform(mus_range["ijv_n"][0], mus_range["ijv_n"][1])
        
        cca_musp = random.uniform(mus_range["cca_musp"][0], mus_range["cca_musp"][1])
        cca_bmie = random.uniform(mus_range["cca_bmie"][0], mus_range["cca_bmie"][1])
        cca_g = random.uniform(mus_range["cca_g"][0], mus_range["cca_g"][1])
        cca_n = random.uniform(mus_range["cca_n"][0], mus_range["cca_n"][1])
                
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

        inp["skin"]["muspx"] = skin_musp
        inp["skin"]["bmie"] = skin_bmie
        inp["skin"]["g"] = skin_g
        inp["skin"]["n"] = skin_n

        inp["fat"]["muspx"] = fat_musp
        inp["fat"]["bmie"] = fat_bmie
        inp["fat"]["g"] = fat_g
        inp["fat"]["n"] = fat_n

        inp["muscle"]["muspx"] = muscle_musp
        inp["muscle"]["bmie"] = muscle_bmie
        inp["muscle"]["g"] = muscle_g
        inp["muscle"]["n"] = muscle_n

        inp["IJV"]["muspx"] = ijv_musp
        inp["IJV"]["bmie"] = ijv_bmie
        inp["IJV"]["g"] = ijv_g
        inp["IJV"]["n"] = ijv_n

        inp["CCA"]["muspx"] = cca_musp
        inp["CCA"]["bmie"] = cca_bmie
        inp["CCA"]["g"] = cca_g
        inp["CCA"]["n"] = cca_n

        inp["geometry"]["skin_thickness"] = geo_skin
        inp["geometry"]["fat_thickness"] = geo_fat
        inp["geometry"]["ijv_radius"] = geo_ijvr
        inp["geometry"]["ijv_depth"] = geo_ijvd
        inp["geometry"]["cca_radius"] = geo_ccar
        inp["geometry"]["cca_depth"] = geo_ccad
        inp["geometry"]["ijv_cca_distance"] = geo_ijvcca
        inp["boundary"]["x_size"] = 150
        inp["boundary"]["y_size"] = 150
        inp["boundary"]["z_size"] = 200


        # make input
        wl = [i for i in range(660, 851)]
        wl_idx = random.sample([ i for i in range(len(wl))], 1)[0]
        wl = wl[wl_idx]
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
        skin_mus = self._calculate_mus(
            wl_idx,
            inp["skin"]["muspx"], 
            inp["skin"]["bmie"],
            inp["skin"]["g"]
        )
        mcx_input["Domain"]["Media"][1]["name"] = "skin"
        mcx_input["Domain"]["Media"][1]["mua"] = 0
        mcx_input["Domain"]["Media"][1]["mus"] = skin_mus
        mcx_input["Domain"]["Media"][1]["g"] = inp["skin"]["g"]
        mcx_input["Domain"]["Media"][1]["n"] = inp["skin"]["n"]

        # fat
        fat_mus = self._calculate_mus(
            wl_idx,
            inp["fat"]["muspx"], 
            inp["fat"]["bmie"],
            inp["fat"]["g"]
        )
        mcx_input["Domain"]["Media"][2]["name"] = "fat"
        mcx_input["Domain"]["Media"][2]["mua"] = 0
        mcx_input["Domain"]["Media"][2]["mus"] = fat_mus
        mcx_input["Domain"]["Media"][2]["g"] = inp["fat"]["g"]
        mcx_input["Domain"]["Media"][2]["n"] = inp["fat"]["n"]

        # muscle
        muscle_mus = self._calculate_mus(
            wl_idx,
            inp["muscle"]["muspx"], 
            inp["muscle"]["bmie"],
            inp["muscle"]["g"]
        )
        mcx_input["Domain"]["Media"][3]["name"] = "muscle"
        mcx_input["Domain"]["Media"][3]["mua"] = 0
        mcx_input["Domain"]["Media"][3]["mus"] = muscle_mus
        mcx_input["Domain"]["Media"][3]["g"] = inp["muscle"]["g"]
        mcx_input["Domain"]["Media"][3]["n"] = inp["muscle"]["n"]
        
        # IJV
        ijv_mus = self._calculate_mus(
            wl_idx,
            inp["IJV"]["muspx"], 
            inp["IJV"]["bmie"],
            inp["IJV"]["g"]
        )
        mcx_input["Domain"]["Media"][4]["name"] = "IJV"
        mcx_input["Domain"]["Media"][4]["mua"] = 0    # for white MC
        mcx_input["Domain"]["Media"][4]["mus"] = ijv_mus 
        mcx_input["Domain"]["Media"][4]["g"] = inp["IJV"]["g"]
        mcx_input["Domain"]["Media"][4]["n"] = inp["IJV"]["n"]
        
        # CCA
        cca_mus = self._calculate_mus(
            wl_idx,
            inp["CCA"]["muspx"], 
            inp["CCA"]["bmie"],
            inp["CCA"]["g"]
        )
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
        with open("train/input_mcx.json", 'w+') as f:
            json.dump(mcx_input, f, indent=4)

        command = self._get_command(idx)
        print(command)
        os.chdir("mcx/bin")
        os.system(command)
        os.chdir("../..")

        # run WMC
        num = 1000
        args = self._get_args(mua_range, num)
        # [吸收數, SDS]
        spec = self.mch.run_wmc_single(os.path.join("train", "mch", "{}.mch".format(idx)), args)

        # mua
        oxy = self.mch.mua['oxy'].values
        deoxy = self.mch.mua['deoxy'].values
        water = self.mch.mua['water'].values
        collagen = self.mch.mua['collagen'].values
        fat = self.mch.mua['fat'].values
        melanin = self.mch.mua['mel'].values
        wavelength = self.mch.mua['wavelength'].values

        # interpolation
        oxy = np.interp(wl, wavelength, oxy)
        deoxy = np.interp(wl, wavelength, deoxy)
        water = np.interp(wl, wavelength, water)
        collagen = np.interp(wl, wavelength, collagen)
        fat = np.interp(wl, wavelength, fat)
        melanin = np.interp(wl, wavelength, melanin)

        # turn the unit 1/cm --> 1/mm
        oxy *= 0.1
        deoxy *= 0.1
        water *= 0.1
        collagen *= 0.1
        fat *= 0.1
        melanin *= 0.1



        for i in range(num):
            skin_mua = self.mch._calculate_mua(
                args[i]["skin"]["blood_volume_fraction"],
                args[i]["skin"]["ScvO2"],
                args[i]["skin"]["water_volume"],
                args[i]["skin"]["fat_volume"],
                args[i]["skin"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
            )
            fat_mua = self.mch._calculate_mua(
                args[i]["fat"]["blood_volume_fraction"],
                args[i]["fat"]["ScvO2"],
                args[i]["fat"]["water_volume"],
                args[i]["fat"]["fat_volume"],
                args[i]["fat"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
            )
            muscle_mua = self.mch._calculate_mua(
                args[i]["muscle"]["blood_volume_fraction"],
                args[i]["muscle"]["ScvO2"],
                args[i]["muscle"]["water_volume"],
                args[i]["muscle"]["fat_volume"],
                args[i]["muscle"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
            )
            ijv_mua = self.mch._calculate_mua(
                args[i]["ijv"]["blood_volume_fraction"],
                args[i]["ijv"]["ScvO2"],
                args[i]["ijv"]["water_volume"],
                args[i]["ijv"]["fat_volume"],
                args[i]["ijv"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
            )
            cca_mua = self.mch._calculate_mua(
                args[i]["cca"]["blood_volume_fraction"],
                args[i]["cca"]["ScvO2"],
                args[i]["cca"]["water_volume"],
                args[i]["cca"]["fat_volume"],
                args[i]["cca"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
            )
            field = "idx, "
            field += "skin_mua, skin_mus, skin_g, skin_n, "
            field += "fat_mua, fat_mus, fat_g, fat_n, "
            field += "muscle_mua, muscle_mus, muscle_g, muscle_n, "
            field += "ijv_mua, ijv_mus, ijv_g, ijv_n, "
            field += "cca_mua, cca_mus, cca_g, cca_n, "
            field += "skin_thickness, fat_thickness, ijv_radius, ijv_depth, cca_radius, cca_depth, ijv_cca_distance, "
            field += "reflectance_20, reflectance_24, reflectance_28"

            values = "'{}', ".format(idx)
            values += "'{}', '{}', '{}', '{}', ".format(skin_mua, skin_mus, skin_g, skin_n)
            values += "'{}', '{}', '{}', '{}', ".format(fat_mua, fat_mus, fat_g, fat_n)
            values += "'{}', '{}', '{}', '{}', ".format(muscle_mua, muscle_mus, muscle_g, muscle_n)
            values += "'{}', '{}', '{}', '{}', ".format(ijv_mua, ijv_mus, ijv_g, ijv_n)
            values += "'{}', '{}', '{}', '{}', ".format(cca_mua, cca_mus, cca_g, cca_n)
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

            sql = "INSERT INTO ijv_ann({}) VALUES({})".format(field, values)
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

    def _get_command(self, idx):
        # create the command for mcx

        session_name = "\"{}\" ".format(idx)
        geometry_file = "\"{}\" ".format(os.path.abspath(os.path.join("train", "input_mcx.json")))
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



    @staticmethod
    def _get_args(x_range, num=50000):
        args = []

        for i in range(num):
            sb = random.uniform(x_range["skin_b"][0], x_range["skin_b"][1])
            ss = random.uniform(x_range["skin_s"][0], x_range["skin_s"][1])
            sw = min(random.uniform(x_range["skin_w"][0], x_range["skin_w"][1]), 1-sb)
            sf = min(random.uniform(x_range["skin_f"][0], x_range["skin_f"][1]), 1-sb-sw)
            sm = 1-sb-sw-sf
            fb = random.uniform(x_range["fat_b"][0], x_range["fat_b"][1])
            fs = random.uniform(x_range["fat_s"][0], x_range["fat_s"][1])
            ff = 1 - fb


            mb = random.uniform(x_range["muscle_b"][0], x_range["muscle_b"][1])
            ms = random.uniform(x_range["muscle_s"][0], x_range["muscle_s"][1])
            mw = min(random.uniform(x_range["muscle_w"][0], x_range["muscle_w"][1]), 1-mb)
            mc = 1 - mb - mw
            is_ = random.uniform(x_range["ijv_s"][0], x_range["ijv_s"][1])
            cs = random.uniform(x_range["cca_s"][0], x_range["cca_s"][1])
            arg = {
                "skin":{
                    "blood_volume_fraction": sb,
                    "ScvO2": ss,
                    "water_volume": sw,
                    "fat_volume": sf,
                    "melanin_volume": sm,
                    "collagen_colume": 0
                },

                "fat":{
                    "blood_volume_fraction": fb,
                    "ScvO2": fs,
                    "water_volume": 0,
                    "fat_volume": ff,
                    "melanin_volume": 0,
                    "collagen_colume": 0
                },

                "muscle":{
                    "blood_volume_fraction": mb,
                    "ScvO2": ms,
                    "water_volume": mw,
                    "fat_volume": 0,
                    "melanin_volume": 0,
                    "collagen_colume": mc
                },
                "ijv":{
                    "blood_volume_fraction": 1,
                    "ScvO2": is_,
                    "water_volume": 0,
                    "fat_volume": 0,
                    "melanin_volume": 0,
                    "collagen_colume": 0
                },
                "cca":{
                    "blood_volume_fraction": 1,
                    "ScvO2": cs,
                    "water_volume": 0,
                    "fat_volume": 0,
                    "melanin_volume": 0,
                    "collagen_colume": 0
                },
            }
            args += [arg]
        
        return args


if __name__ == "__main__":
    import sys
    mcx = MCXGen()
    for i in range(int(sys.argv[1])):
        mcx.train()