import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from glob import glob

from kde import smooth

date = "20190619"
path = "data/raw/" + date

calib_wl = np.loadtxt(path + "/calib_wl.csv", delimiter=",")
bg1 = np.loadtxt(path + "/background.csv", delimiter=",")
bg2 = np.loadtxt(path + "/background_2.csv", delimiter=",")

path_list = glob(path + "/live/live_*_*.csv")
path_list.sort(key=lambda x: (int(x.split('/')[-1].split('_')[-2]), int(x.split('/')[-1].split('_')[-1].strip('.csv'))))

wl = [i for i in range(660, 851, 10)]


# live
for i, p in enumerate(path_list):
    l = np.loadtxt(p, delimiter=",")
    l -= bg1
    l = l.mean(0)

    l = np.interp(wl, calib_wl, l)

    df = {}

    if i % 2 == 0:
        df["wavelength"] = wl
        df["max"] = smooth(l)
    else:
        df["min"] = smooth(l)
        df = pd.DataFrame(df)
        df = df[["wavelength", "max", "min"]]
        df.to_csv("data/processed/" + date + "/IJV/live/" + date + str(i//2+1) + ".csv", index=None)

# phantom
path_list = path + "/phantom/phantom_{}.csv" 
df_dict = {"wavelength": wl}
for i, p in enumerate("chiken"):
    pp = np.loadtxt(path_list.format(p), delimiter=",")
    if p in "chi":
        pp -= bg1
    else:
        pp -= bg2

    pp = np.interp(wl, calib_wl, smooth(pp))
    df_dict[p] = pp

df_dict = df_dict[["wavelength", "c", "h", "i", "k", "e", "n"]]
df_dict.to_csv("data/processed/" + date + "/IJV/phantom/" + date + ".csv", index=None)
