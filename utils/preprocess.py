import os
import pandas as pd 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks_cwt
from calibration import Calibrator


def preprocess_phantom(input_path, result_path="result"):
    
    # setting
    bg_path = os.path.join(input_path, "background.csv")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    phantom_path = os.path.join(input_path, "phantom")
    wl = [i for i in range(650, 1001, 10)]



    bg = np.loadtxt(bg_path, delimiter=",")
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')

    phantom_list = glob(os.path.join(phantom_path, "phantom_*.csv"))
    phantom_list.sort(key=lambda x: x[-5])

    data = []
    for p in phantom_list:
        data.append(np.loadtxt(p, delimiter=","))
    data = np.asarray(data)

    data_sub_bg = data.mean(1) - bg.mean(0)

    data_interp = []
    for d in data_sub_bg:
        data_interp.append(np.interp(wl, calib_wl, d))
    data_interp = np.asarray(data_interp)

    # plot and
    # save
    df_dict = {}
    df_dict['wavelength'] = wl
    for p, d in zip(phantom_list, data_interp):
        df_dict[p[-5]] = d
        plt.plot(wl, d, label=p[-5])
    plt.grid()
    plt.legend()
    plt.xlabel("wavelength[nm]")
    plt.ylabel("reflectance[-]")
    plt.savefig(os.path.join(result_path, "phantom_" + input_path + ".png"))
    plt.clf()

    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(result_path, "phantom_" + input_path + ".csv"), index=None)

    return data_interp


def preprocess_live(input_path, result_path="result", calibrate=True):

    # setting
    bg_path = os.path.join(input_path, "background.csv")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    live_path = os.path.join(input_path, "live")
    wl = [i for i in range(650, 1001, 10)]

    bg = np.loadtxt(bg_path, delimiter=",")
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')


    live_list = glob(os.path.join(live_path, "live_*_*.csv"))
    live_list.sort(key=lambda x: (x.split('_')[1], x.split('_')[2].strip(".csv")))
    live_idx = [i.split("/")[-1].strip("live_").strip(".csv") for i in live_list]

    if calibrate:
        calib = Calibrator()
        calib.fit(measured=0, simulated=0)
        
    for l, n in zip(live_list, live_idx):
        live = np.loadtxt(l, delimiter=",")

        live -= bg.mean(0)

        live_interp = []
        for l in live:
            live_interp.append(np.interp(wl, calib_wl, l))
        live_interp = np.asarray(live_interp)

        live_crop = live[:, (calib_wl > 600) & (calib_wl < 950)]
        live_crop = live_crop.mean(1)
        live_crop = 1 - live_crop/65535

        max_index = find_peaks_cwt(live_crop, np.arange(1, 20))
        min_index = find_peaks_cwt(1-live_crop, np.arange(1, 20))

        plt.figure()
        plt.plot(live_crop)
        plt.scatter(max_index, live_crop[max_index], label='max')
        plt.scatter(min_index, live_crop[min_index], label='min')
        plt.legend()
        plt.grid()
        plt.xlabel("time[frame]")
        plt.ylabel("reflectance[-]")
        plt.savefig(os.path.join(result_path, "live_" + input_path + "_" + n + "_peak.png"))
        plt.clf()

        live_max = live[max_index].mean(0)
        live_min = live[min_index].mean(0)

        live_max_interp = np.interp(wl, calib_wl, live_max)
        live_min_interp = np.interp(wl, calib_wl, live_min)

        if calibrate:
            live_max_interp = calib.calibrate(live_max_interp)
            live_min_interp = calib.calibrate(live_min_interp)

        plt.figure()
        plt.plot(wl, live_max_interp, label="max")
        plt.plot(wl, live_min_interp, label="min")
        plt.legend()
        plt.grid()
        plt.xlabel("wavelength[nm]")
        plt.ylabel("reflectance[-]")
        plt.savefig(os.path.join(result_path, "live_" + input_path + "_" + n + ".png"))
        plt.clf()

        df_dict = {}
        df_dict["wavelength"] = wl
        df_dict["max"] = live_max_interp 
        df_dict["min"] = live_min_interp

        df = pd.DataFrame(df_dict)

        df.to_csv(os.path.join(result_path, "live_" + input_path + "_" + n + ".csv"))






