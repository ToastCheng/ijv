import json
from time import time
from glob import glob
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 

from utils import load_mch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MCHHandler:
    # This class is a .mch file handler.

    def __init__(
        self,
        config="config.json"
        ):

        with open(config) as f:
            self.config = json.loads(f.read())

        mch_file_path = os.path.join("output", self.config["session_id"], "mcx_output")
        self.mch_list = glob(os.path.join(mch_file_path, "*.mch"))
        self.mch_list.sort(key=lambda x: int(x.split("_")[-2]))

        self.mua = pd.read_csv(self.config["coefficients"])
        self.wavelength = pd.read_csv(self.config["wavelength"])
        with open(self.config["parameters"]) as f: 
            self.input = json.loads(f.read())

        self.detector_na = self.config["detector_na"]
        self.detector_n = self.config["medium_n"]
        self.critical_angle = np.arcsin(self.detector_na/self.detector_n)

    def calculate_reflectance_white(self, args):

        spectra = []
        portions = []

        for f in self.mch_list:
            wl = int(f[-12:-9])
            df, header, photon = self._load_mch(f)

            df = df[np.arccos(df.angle.abs()) <= self.critical_angle]
            if len(df) == 0:
                print('no photon detected at %d' % wl)
                return None, None
            df = df.reset_index(drop=True)

            # [photon, medium]
            path_length = df.iloc[:, 1:-1].values

            path_length = torch.tensor(path_length).float().to(device)

            # [medium, ScvO2]
            mua = self._make_tissue_white(wl, header, args)
            mua = torch.tensor(mua).float().to(device)

            # [photon, ScvO2]
            weight = torch.exp(-torch.matmul(path_length, mua) *header["unitmm"]) 
            
            # NEW 
            # df["weight"] = weight.tolist()
            # 3 計算實際路徑長*權重 (in 95%)
            # print(path_length.shape)
            # print(weight[:, 96].shape)
            path_length = (path_length * weight[:].unsqueeze(1))
            skin_portion = (path_length.mean(0)[0] ).tolist()
            fat_portion = (path_length.mean(0)[1]).tolist()
            muscle_portion = (path_length.mean(0)[2]).tolist()
            ijv_portion = (path_length.mean(0)[3]).tolist()
            cca_portion = (path_length.mean(0)[4]).tolist()


            # [1, ScvO2]
            result = torch.zeros(1).float().to(device)

            # seperate photon with different detector
            for idx in range(1, header["detnum"]+1):

                # get the index of specific index
                detector_list = df.index[df["detector_idx"] == idx].tolist()
                if len(detector_list) == 0:
                    # print("detector #%d detected no photon" % idx)
                    result = torch.cat((result, torch.zeros(1, weight.shape[1]).float().to(device)), 0)
                    continue
                # pick the photon that detected by specific detector
                
                # [photon, ScvO2]
                _weight = weight[detector_list]
                # print(_weight.shape)
                _weight = _weight.sum(0)
                _weight = _weight.unsqueeze(0)
                # print(_weight.shape)
                # print(result.shape)
                result = torch.cat((result, _weight), 0)

            # [SDS, ScvO2]
            result = result[1:]

            s = float(result.cpu().numpy()/header["total_photon"])
            
            spectra.append(s)
            portions.append(
                (skin_portion, fat_portion, muscle_portion, ijv_portion, cca_portion)
                )

        return spectra, portions
            

    def _make_tissue_white(self, wl, header, args):
        
        # mua
        oxy = self.mua['oxy'].values
        deoxy = self.mua['deoxy'].values
        water = self.mua['water'].values
        collagen = self.mua['collagen'].values
        fat = self.mua['fat'].values
        melanin = self.mua['mel'].values
        wavelength = self.mua['wavelength'].values

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

        # [medium, 1]
        mua = np.zeros((header["maxmedia"], 1))



        skin = self._calculate_mua(
            args["skin"]["blood_volume_fraction"],
            args["skin"]["ScvO2"],
            args["skin"]["water_volume"],
            args["skin"]["fat_volume"],
            args["skin"]["melanin_volume"],
            oxy, 
            deoxy, 
            water,
            fat,
            melanin
            )

        fat = self._calculate_mua(
            args["fat"]["blood_volume_fraction"],
            args["fat"]["ScvO2"],
            args["fat"]["water_volume"],
            args["fat"]["fat_volume"],
            args["fat"]["melanin_volume"],
            oxy, 
            deoxy, 
            water,
            fat,
            melanin
            )

        muscle = self._calculate_muscle_mua(
            args["muscle"]["water_volume"],
            water,
            collagen
            )

        IJV = self._calculate_mua(
            args["IJV"]["blood_volume_fraction"],
            args["IJV"]["ScvO2"],
            args["IJV"]["water_volume"],
            args["IJV"]["fat_volume"],
            args["IJV"]["melanin_volume"],
            oxy, 
            deoxy, 
            water,
            fat,
            melanin
            )
        CCA = self._calculate_mua(
            args["CCA"]["blood_volume_fraction"],
            args["CCA"]["ScvO2"],
            args["CCA"]["water_volume"],
            args["CCA"]["fat_volume"],
            args["CCA"]["melanin_volume"],
            oxy, 
            deoxy, 
            water,
            fat,
            melanin
            )

        _mua = np.concatenate(
            [np.expand_dims(skin, 0),
            np.expand_dims(fat, 0),
             np.expand_dims(muscle, 0), 
             np.expand_dims(IJV, 0), 
             np.expand_dims(CCA, 0)], 0
             )

        return _mua


    @staticmethod
    def _load_mch(path):
        data = load_mch(path)
        # check if the mcx saved the photon seed
        if data[1]["seed_byte"] == 0:
            df, header = data
            photon_seed = None
        elif data[1]["seed_byte"] == 1:
            df, header, photon_seed = data

        num_media = header["maxmedia"] 
        # selected_list = [0] + [i for i in range(num_media+1, 2*num_media+1)] + [-1]
        selected_list = [0] + [i for i in range(2, 2 + num_media)] + [-1]
        df = df[:, selected_list]
        label = ["detector_idx"]
        label += ["media_{}".format(i) for i in range(header["maxmedia"])]
        label += ["angle"]
        df = pd.DataFrame(df, columns=label)
        

        return df, header, photon_seed



    @staticmethod
    def _calculate_mua(b, s, w, f, m, oxy, deoxy, water, fat, melanin):
        mua = b * (s * oxy + (1-s) * deoxy) + w * water + f * fat + m * melanin
        return mua

    @staticmethod
    def _calculate_muscle_mua(w, water, collagen):
        mua = w * water + (1-w) * collagen
        return mua 




