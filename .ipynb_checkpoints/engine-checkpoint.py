import torch 
import numpy as np 
import pandas as pd 
from model.model import Model

args_template = {
    "skin":{
        "blood_volume_fraction": 0.5,
        "ScvO2": 0.7,
        "water_volume": 0,
        "fat_volume": 0,
        "melanin_volume": 0.01,
        "n": 1.40,
        "g": 0.71,
        "muspx": 1,
        "bmie": 0.453
    },

    "fat":{
        "blood_volume_fraction": 0,
        "ScvO2": 0,
        "water_volume": 0,
        "fat_volume": 1,
        "melanin_volume": 0,
        "n": 1.40,
        "g": 0.90,
        "muspx": 1,
        "bmie": 8
    },

    "muscle":{
        "blood_volume_fraction": 0.05,
        "ScvO2": 0.80,
        "water_volume": 0.75,
        "fat_volume": 0,
        "melanin_volume": 0,
        "n": 1.40,
        "g": 0.9,
        "muspx": 0.1,
        "bmie": 0.52
    },

    "IJV":{
        "blood_volume_fraction": 1.0,
        "ScvO2": 0.70,
        "water_volume": 0,
        "fat_volume": 0,
        "melanin_volume": 0,
        "n": 1.40,
        "g": 0.90,
        "muspx": 1.0,
        "bmie": 0.7
    },

    "CCA":{
        "blood_volume_fraction": 1.0,
        "ScvO2": 0.98,
        "water_volume": 0,
        "fat_volume": 0,
        "melanin_volume": 0,
        "n": 1.40,
        "g": 0.94,
        "muspx": 10.0,
        "bmie": 1.0
    },

    # skin_thickness, fat_thickness, ijv_radius, ijv_depth, cca_radius, cca_depth, ijv_cca_distance
    "geometry": [1.23, 1.64, 5.96, 12.98, 2.75, 21.98, 0.91],

}

class Engine:
    def __init__(self, path, phantom=False):
        self.model = Model()
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        self.phantom = phantom

        # absorption
        self.wl = np.array([i for i in range(660, 811, 10)])
 
        self.mua = pd.read_csv("input/coefficients.csv")
        oxy = self.mua["oxy"].values
        deoxy = self.mua['deoxy'].values
        water = self.mua['water'].values
        collagen = self.mua['collagen'].values
        fat = self.mua['fat'].values
        melanin = self.mua['mel'].values
        wl = self.mua['wavelength'].values

        # interpolation
        oxy = np.interp(self.wl, wl, oxy)
        deoxy = np.interp(self.wl, wl, deoxy)
        collagen = np.interp(self.wl, wl, collagen)
        water = np.interp(self.wl, wl, water)
        fat = np.interp(self.wl, wl, fat)
        melanin = np.interp(self.wl, wl, melanin)

        # 1/cm --> 1/mm
        self.oxy = oxy * 0.1
        self.deoxy = deoxy * 0.1
        self.water = water * 0.1
        self.collagen = collagen * 0.1
        self.fat = fat * 0.1
        self.melanin = melanin * 0.1

        if self.phantom:
            mua = pd.read_csv("phantom/tissue_mua.csv")
            mus = pd.read_csv("phantom/tissue_mus.csv")
#             ink = pd.read_csv("phantom/ink.csv")
            ink = pd.read_csv("phantom/ink_20190805.csv")

            mua = mua[
                (mua["wl"] >= self.wl[0])&
                (mua["wl"] <= self.wl[-1])
            ]
            mus = mus[
                (mus["wl"] >= self.wl[0])&
                (mus["wl"] <= self.wl[-1])
            ]

            self.skin_mua = np.interp(self.wl, mua["wl"], mua["skin"])
            self.fat_mua = np.interp(self.wl, mua["wl"], mua["fat"])
            self.muscle_mua = np.interp(self.wl, mua["wl"], mua["ml"])
            self.ink780 = np.interp(self.wl, ink["wl"], ink["780"])
            self.ink832 = np.interp(self.wl, ink["wl"], ink["832"])

            self.skin_mus = np.interp(self.wl, mus["wl"], mus["skin"])
            self.fat_mus = np.interp(self.wl, mus["wl"], mus["fat"])
            self.muscle_mus = np.interp(self.wl, mus["wl"], mus["ml"])


    def get_spectrum(self, args, warning=False):

        geo = args["geometry"]

        spec = []
        for idx in range(len(self.wl)):
            param = self._make_param(args, idx, warning)

            s = self._predict(param, geo)
            spec += [float(s[0][0])]

        return np.array(spec)


    def _predict(self, param, geo):
        if len(param.shape) == 1:
            param = torch.tensor(param).float().unsqueeze(0)
            geo = torch.tensor(geo).float().unsqueeze(0)
        elif len(param.shape) == 2:
            param = torch.tensor(param).float()
            geo = torch.tensor(geo).float()

        pred = self.model(param, geo)
        pred = torch.exp(pred)

        return pred

    #deprecated
    def _make_param_tissue_phantom(self, args, idx):

        ratio = args["780ratio"]
        skin_mua = self.skin_mua[idx]
        skin_mus = self.skin_mus[idx]
        skin_g = 0
        skin_n = 1.4
        
        fat_mua = self.fat_mua[idx]
        fat_mus = self.fat_mus[idx]
        fat_g = 0
        fat_n = 1.4

        muscle_mua = self.muscle_mua[idx]
        muscle_mus = self.muscle_mus[idx]
        muscle_g = 0
        muscle_n = 1.4

        ijv_mua = self.ink832[idx] * (1-ratio) + self.ink780[idx] * ratio
        ijv_mus = 1
        ijv_g = 0
        ijv_n = 1.4

        cca_mua = self.muscle_mua[idx]
        cca_mus = self.muscle_mus[idx]
        cca_g = 0
        cca_n = 1.4

        return np.array([
            skin_mua, skin_mus, skin_g, skin_n,
            fat_mua, fat_mus, fat_g, fat_n,
            muscle_mua, muscle_mus, muscle_g, muscle_n,
            ijv_mua, ijv_mus, ijv_g, ijv_n,
            cca_mua, cca_mus, cca_g, cca_n,
        ])

    def _make_param(self, hyper_param, idx, warning=False):
        skin = hyper_param["skin"]
        fat = hyper_param["fat"]
        muscle = hyper_param["muscle"]
        ijv = hyper_param["IJV"]
        cca = hyper_param["CCA"]


        if self.phantom:
            skin_mua = self._mua_phantom_pdms(skin["water_volume"], idx)
            fat_mua = self._mua_phantom_pdms(fat["ScvO2"], idx)
            muscle_mua = self._mua_phantom_pdms(muscle["water_volume"], idx)
            ijv_mua = self._mua_phantom_ink(ijv, idx)
            cca_mua = muscle_mua

        else:
            skin_mua = self._mua(skin, idx)
            fat_mua = self._mua(fat, idx)
            muscle_mua = self._mua(muscle, idx)
            ijv_mua = self._mua(ijv, idx)
            cca_mua = self._mua(cca, idx)

        skin_mus = self._mus(skin, idx)
        fat_mus = self._mus(fat, idx)
        muscle_mus = self._mus(muscle, idx)
        ijv_mus = self._mus(ijv, idx)
        cca_mus = self._mus(cca, idx)
        
        if warning:
#             if skin_mus > 5:
#                 print(f"[Warning]: skin mus is high: {skin_mus}")
#             if fat_mus > 5:
#                 print(f"[Warning]: fat mus is high: {fat_mus}")
#             if muscle_mus > 5:
#                 print(f"[Warning]: muscle mus is high: {muscle_mus}")
#             if ijv_mus > 5:
#                 print(f"[Warning]: ijv mus is high: {ijv_mus}")
#             if cca_mus > 5:
#                 print(f"[Warning]: cca mus is high: {cca_mus}")
            if skin_mus > 8:
                print(f"[Warning]: skin mus is high: {skin_mus}")
            if fat_mus > 10:
                print(f"[Warning]: fat mus is high: {fat_mus}")
            if muscle_mus > 10:
                print(f"[Warning]: muscle mus is high: {muscle_mus}")
            if ijv_mus > 5:
                print(f"[Warning]: ijv mus is high: {ijv_mus}")
            if cca_mus > 5:
                print(f"[Warning]: cca mus is high: {cca_mus}")
        

        return np.array([
            skin_mua, skin_mus, skin["g"], skin["n"],
            fat_mua, fat_mus, fat["g"], fat["n"],
            muscle_mua, muscle_mus, muscle["g"], muscle["n"],
            ijv_mua, ijv_mus, ijv["g"], ijv["n"],
            cca_mua, cca_mus, cca["g"], cca["n"],
        ])



    def _mua(self, medium, idx):
        b = medium["blood_volume_fraction"]
        s = medium["ScvO2"]
        w = medium["water_volume"]
        f = medium["fat_volume"]
        m = medium["melanin_volume"]
        c = medium["collagen_volume"]

        mua = b * (s * self.oxy[idx] + (1-s) * self.deoxy[idx]) + w * self.water[idx] + f * self.fat[idx] + m * self.melanin[idx] + c * self.collagen[idx]
        return mua

    def _mua_phantom_ink(self, medium, idx):
        r = medium["ScvO2"]
        mua = self.ink780[idx] * r + self.ink832[idx] * (1-r)
        return mua

    def _mua_phantom_pdms(self, c, idx):
        mua = c * 10 * self.skin_mua[idx]
        return mua

    def _mus(self, medium, idx):
        muspx = medium["muspx"]
        bmie = medium["bmie"]
        g = medium["g"] 

        mus_p = muspx * (self.wl[idx]/500) ** (-bmie)

        mus = mus_p/(1-g) * 0.1
        
        return mus 







if __name__ == "__main__":
    engine = Engine()
