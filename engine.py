import torch 
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
    def __init__(self, path):
        self.model = Model()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # absorption
        self.wl = np.array([i for i in range(660, 851)])
 
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

    def get_spectrum(self, args):

        geo = args["geometry"]
        hyper_param = args["hyper_param"]

        spec = []
        for idx in range(len(self.wl)):
            param = self._make_param(hyper_param, idx)
            s = self._predict(param, geo)
            spec += [s]

        return spec 


    def _predict(self, param, geo):
        pred = model(param, geo)
        pred = torch.exp(pred)

        return pred

    def _make_param(self, hyper_param, idx):
        skin = hyper_param["skin"]
        fat = hyper_param["fat"]
        muscle = hyper_param["muscle"]
        ijv = hyper_param["IJV"]
        cca = hyper_param["CCA"]

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

        mua = b * (s * self.oxy[idx] + (1-s) * self.deoxy[idx]) + w * self.water[idx] + f * self.fat[idx] + m * self.melanin[idx]
        return mua



    def _mus(self, medium, idx):
        muspx = medium["muspx"]
        bmie = medium["bmie"]
        g = medium["g"] 

        mus_p = muspx500 * (self.wl[idx]/500) ** (-bmie)

        mus = mus_p/(1-g) * 0.1
        
        return mus 







if __name__ == "__main__":
    engine = Engine()
