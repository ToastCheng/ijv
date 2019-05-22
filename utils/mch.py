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
# device = torch.device('cpu')

class MCHHandler:
    # This class is a .mch file handler.

    def __init__(self, config=None):
        
        if config is not None:
            self.load_config(config)
        else:
            self.config = None


    def load_config(self, config):
        with open(config) as f:
            self.config = json.loads(f.read())

        mch_file_path = os.path.join("output", self.config["session_id"], "mcx_output")
        self.mch_list = glob(os.path.join(mch_file_path, "*.mch"))
        if self.config["type"] == "ijv":
            self.mch_list.sort(key=lambda x: int(x.split("_")[-1].strip('.mch')))
        elif self.config["type"] == "muscle":
            self.mch_list.sort(key=lambda x: int(x.split("_")[-1].strip('.mch')))
        elif self.config["type"] == "phantom":
            pid = { c: i for i, c in enumerate("CHIKEN")}
            self.mch_list.sort(
                key=lambda x: (int(x.split("_")[-2]), pid[x.split('_')[-1].strip('.mch')])
            )

        if self.config["type"] != "phantom":
            self.mua = pd.read_csv(self.config["coefficients"])
        else:
            self.mua = pd.read_csv(self.config["phantom_mua"])


        self.detector_na = self.config["detector_na"]
        self.detector_n = self.config["medium_n"]
        self.critical_angle = np.arcsin(self.detector_na/self.detector_n)     

    def quick_load_ijv(self, mch_file_path):
        self.config = {
            "type": "ijv"
        }

        self.mch_list = glob(os.path.join(mch_file_path, "*.mch"))
        self.mch_list.sort(key=lambda x: int(x.split("_")[-1].strip('.mch')))

        self.mua = pd.read_csv("input/coefficients.csv")
        self.detector_na = 0.12
        self.detector_n = 1.4
        self.critical_angle = np.arcsin(self.detector_na/self.detector_n)     


        
    def _make_tissue_phantom(self, wl, header, phantom_id):

        phantom = self.mua[phantom_id].values
        wavelength = self.mua['wl'].values

        phantom = np.interp([wl], wavelength, phantom)

        # if the mua is 1/cm, you need to change the unit into 1/mm
        # mua *= 0.1

        return np.expand_dims(phantom, 0)


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
        # mua = np.zeros((header["maxmedia"], 1))

        _mua = []
        for arg in args:
            skin = self._calculate_mua(
                arg["skin"]["blood_volume_fraction"],
                arg["skin"]["ScvO2"],
                arg["skin"]["water_volume"],
                arg["skin"]["fat_volume"],
                arg["skin"]["melanin_volume"],
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
                )

            fat = self._calculate_mua(
                0,
                0,
                0,
                arg["fat"]["fat_volume"],
                0,
                oxy, 
                deoxy, 
                water,
                fat,
                melanin,
                collagen
                )

            muscle = self._calculate_mua(
                arg["muscle"]["blood_volume_fraction"],
                arg["muscle"]["ScvO2"],
                arg["muscle"]["water_volume"],
                arg["muscle"]["fat_volume"],
                arg["muscle"]["melanin_volume"],
                oxy,
                deoxy,
                water,
                fat,
                melanin,
                collagen,
                )

            if self.config["type"] == "ijv":
                IJV = self._calculate_mua(
                    arg["ijv"]["blood_volume_fraction"],
                    arg["ijv"]["ScvO2"],
                    arg["ijv"]["water_volume"],
                    arg["ijv"]["fat_volume"],
                    arg["ijv"]["melanin_volume"],
                    oxy, 
                    deoxy, 
                    water,
                    fat,
                    melanin,
                    collagen
                    )
                CCA = self._calculate_mua(
                    arg["cca"]["blood_volume_fraction"],
                    arg["cca"]["ScvO2"],
                    arg["cca"]["water_volume"],
                    arg["cca"]["fat_volume"],
                    arg["cca"]["melanin_volume"],
                    oxy, 
                    deoxy, 
                    water,
                    fat,
                    melanin,
                    collagen
                    )

            if self.config["type"] == "ijv":
                _mua += [np.concatenate(
                    [np.expand_dims(skin, 0),
                    np.expand_dims(fat, 0),
                     np.expand_dims(muscle, 0), 
                     np.expand_dims(IJV, 0), 
                     np.expand_dims(CCA, 0)], 0
                     )]

            elif self.config["type"] == "muscle":
                _mua += [np.concatenate(
                    [np.expand_dims(skin, 0),
                     np.expand_dims(fat, 0),
                     np.expand_dims(muscle, 0)
                    ], 0
                     )]

            else:
                raise Exception("tissue type does not supported yet")
                
        return np.asarray(_mua).T



    def run_wmc(self, args=None, prism=False):

        if not self.config:
            raise Exception("Should specify the config file first!")

        if isinstance(args, dict):
            args = [args]
        if args is None:
            args = [0]
            
        spectra = []
        portions = []


        for f in self.mch_list:
            if self.config["type"] == "ijv":
                wl = int(f.split('_')[-1].strip('.mch'))
            elif self.config["type"] == "muscle":
                wl = int(f.split('_')[-1].strip('.mch'))
            elif self.config["type"] == "phantom":
                wl = int(f.split('_')[-2])
            else:
                raise Exception("Tissue type error!")

            df, header, photon = self._load_mch(f)

            df = df[np.arccos(df.angle.abs()) <= self.critical_angle]
            if len(df) == 0:
                print('no photon detected at %d' % wl)
                return None, None
            df = df.reset_index(drop=True)

            # [光子數, 組織數]
            # 0: detector index
            # 1: prism
            if prism:
                path_length = df.iloc[:, 2:-1].values
            else:
                path_length = df.iloc[:, 1:-1].values

            path_length = torch.tensor(path_length).float().to(device)

            # [ 組織數, 吸收數]
            if self.config["type"] != "phantom":
                mua = self._make_tissue_white(wl, header, args)
            else:
                phantom_id = f.split('_')[-1].strip('.mch')
                mua = self._make_tissue_phantom(wl, header, phantom_id)

            mua = torch.tensor(mua).float().to(device)

            
            # [光子數, 吸收數]
            weight = torch.exp(-torch.matmul(path_length, mua) *header["unitmm"]) 


            # NEW 
            # df["weight"] = weight.tolist()
            # 3 計算實際路徑長*權重 (in 95%)

            path_length = (path_length.transpose(0, 1) @ weight)
            if self.config["type"] == "ijv":
                skin_portion = (path_length[0] ).tolist()
                fat_portion = (path_length[1]).tolist()
                muscle_portion = (path_length[2]).tolist()
                ijv_portion = (path_length[3]).tolist()
                cca_portion = (path_length[4]).tolist()

            elif self.config["type"] == "muscle":
                skin_portion = (path_length.mean(0)[0] ).tolist()
                fat_portion = (path_length.mean(0)[1]).tolist()
                muscle_portion = (path_length.mean(0)[2]).tolist()

            elif self.config["type"] == "phantom":
#                 phantom_portion = (path_length.mean(0)[0] ).tolist()
                phantom_portion = [0]
            ###
            ######
            #########

            # [1, 吸收數]
            result = torch.zeros(1, len(args)).float().to(device)

            # seperate photon with different detector
            for idx in range(1, header["detnum"]+1):

                # get the index of specific index
                detector_list = df.index[df["detector_idx"] == idx].tolist()
                if len(detector_list) == 0:
                    # print("detector #%d detected 0 photon" % idx)
                    result = torch.cat((result, torch.zeros(1, weight.shape[1]).float().to(device)), 0)
                    continue
                    
                # pick the photon that detected by specific detector
                
                # [光子數, 吸收數]
                _weight = weight[detector_list]
                
                _weight = _weight.sum(0)
                _weight = _weight.unsqueeze(0)

                
                result = torch.cat((result, _weight), 0)

            # [SDS, ScvO2]
            result = result[1:]
            s = result.cpu().numpy()/header["total_photon"]
            
            spectra.append(s)

            if self.config["type"] == "ijv":
                portions.append(
                    [skin_portion, fat_portion, muscle_portion, ijv_portion, cca_portion]
                )
            elif self.config["type"] == "muscle":
                portions.append(
                    [skin_portion, fat_portion, muscle_portion]
                )
            elif self.config["type"] == "phantom":
                portions.append([phantom_portion])
            else:
                raise Exception("tissue type does not supported yet")

        return np.asarray(spectra).T, np.asarray(portions).T
            





    def _load_mch(self, path):
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
        self.df = df
        

        return df, header, photon_seed


    @staticmethod
    def _calculate_mua(b, s, w, f, m, oxy, deoxy, water, fat, melanin, collagen):
        mua = b * (s * oxy + (1-s) * deoxy) + w * water + f * fat + m * melanin + (1-w-f-m-b) * collagen
        return mua

    @staticmethod
    def _calculate_muscle_mua(b, s, w, oxy, deoxy, water, collagen):
        # to be deprecate
        mua = w * water + (1-w-b) * collagen + b * (s * oxy + (1-s) * deoxy)

        return mua 



if __name__ == "__main__":
    args = {
        "skin":{
            "blood_volume_fraction": 0.5,
            "ScvO2": 0.7,
            "water_volume": 0.49,
            "fat_volume": 0,
            "melanin_volume": 0.01,
        },

        "fat":{
            "blood_volume_fraction": 0,
            "ScvO2": 0,
            "water_volume": 0,
            "fat_volume": 1,
            "melanin_volume": 0,
        },

        "muscle":{
            "blood_volume_fraction": 0.05,
            "ScvO2": 0.80,
            "water_volume": 0.75,
            "fat_volume": 0,
            "melanin_volume": 0,
        },
        "ijv":{
            "blood_volume_fraction": 1,
            "ScvO2": 0.80,
            "water_volume": 0,
            "fat_volume": 0,
            "melanin_volume": 0,
        },
        "cca":{
            "blood_volume_fraction": 1,
            "ScvO2": 0.80,
            "water_volume": 0,
            "fat_volume": 0,
            "melanin_volume": 0,
        },
    }
