import os
import pandas as pd 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks_cwt
from PyEMD import EMD


class Calibrator:
    def __init__(self):
        self.a = []
        self.b = []

    def fit(self, measured, simulated, cross_valid=True):
        """
        [input]
        measured: 
            2D array of measured spectrum
            shape: [num_phantom, num_wavelength]
        simulated:
            2D array of simulated spectrum
            shape: [num_phantom, num_wavelength]
        [output]
        a, b:
            in each wavelength
            simulated = a * measured + b

        """
    
        num_p = measured.shape[0]
        num_wl = measured.shape[1]

        r_square_max = 0
        r_square = []

        # leave one out cross validation
        for one_out in range(-1, num_p):
            index = [i for i in range(num_p) if i != one_out]
            _measured = measured[index]
            _simulated = simulated[index]
            a = []
            b = []

            for m, s in zip(_measured.T, _simulated.T):
                aa, bb = np.polyfit(m, s, 1)
                a.append(aa)
                b.append(bb)

            _r_square = []
            for idx, (x, y) in enumerate(zip(_measured.T, _simulated.T)):

                y_fit = x * a[idx] + b[idx]
                residual = ((y-y_fit)**2).sum()
                SS_total = ((y.mean()-y)**2).sum()
                _r_square.append(1 - residual/SS_total)

            print("leave: %d, r_square: %.2f" % (one_out, np.mean(_r_square)))
            if np.mean(_r_square) > r_square_max:
                self.a = np.asarray(a)
                self.b = np.asarray(b)
                r_square_max = np.mean(_r_square)
                r_square = _r_square.copy()

        return self.a, self.b, r_square

    def calibrate(self, measured):

        measured = np.asarray(measured)
        if len(measured.shape) == 1:
            measured = np.expand_dims(measured, 0)
        for idx, m in enumerate(measured):
            if not m.shape == self.a.shape:
                print("measured shape: ", m.shape)
                print("calibrate shape: ", self.a.shape)
                raise Exception("input shape does not match!")
            measured[idx] = self.a * m + self.b

        return measured


def preprocess_phantom(input_date):
    
    # setting
    input_path = os.path.join("data", "raw", input_date)
    output_path = os.path.join("data", "processed", input_date)

    if not os.path.isdir(input_path):
        raise Exception("There is no raw data with id: {}".format(input_date))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(os.path.join(output_path, "phantom")):
        os.mkdir(os.path.join(output_path, "phantom"))


    bg_path = os.path.join(input_path, "background.csv")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    phantom_path = os.path.join(input_path, "phantom")

    if not os.path.isfile(bg_path):
        raise Exception("background.csv does not exist!")

    if not os.path.isfile(calib_wl_path):
        raise Exception("calib_wl.csv does not exist!")

    if not os.path.isdir(phantom_path):
        raise Exception("Folder phantom does not exist!")

    wl = [i for i in range(650, 1001, 10)]

    bg = np.loadtxt(bg_path, delimiter=",")
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')

    phantom_list = glob(os.path.join(phantom_path, "phantom_*.csv"))
    phantom_list.sort(key=lambda x: x.split('_')[-1].strip('.csv'))

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
    plt.savefig(os.path.join(output_path, "phantom", input_date + ".png"))
    plt.clf()

    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(output_path, "phantom", input_date + ".csv"), index=None)

    return data_interp


def preprocess_live(input_date):

    # setting
    input_path = os.path.join("data", "raw", input_date)
    output_path = os.path.join("data", "processed", input_date)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(os.path.join(output_path, "live")):
        os.mkdir(os.path.join(output_path, "live"))

    bg_path = os.path.join(input_path, "background.csv")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    live_path = os.path.join(input_path, "live")
    wl = [i for i in range(650, 1001, 10)]

    bg = np.loadtxt(bg_path, delimiter=",")
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')


    live_list = glob(os.path.join(live_path, "live_*_*.csv"))
    live_list.sort(key=lambda x: (x.split('_')[1], x.split('_')[2].strip(".csv")))
    live_idx = [i.split("/")[-1].strip("live_").strip(".csv") for i in live_list]

        
    for l, n in zip(live_list, live_idx):
        live = np.loadtxt(l, delimiter=",")

        live -= bg.mean(0)


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
        plt.savefig(os.path.join(output_path, "live", input_date + "_" + n + "_peak.png"))
        plt.clf()

        # remove artifact
        imf = EMD().emd(live_crop, np.arange(len(live_crop)))
        artifact = imf[-1] - imf[-1].mean()
        live_crop = live_crop - artifact


        # 1 frame = 0.05sec
        # So there would be 20 frame for 1 sec!
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
        plt.savefig(os.path.join(output_path, "live", input_date + "_" + n + "_peak_emd.png"))
        plt.clf()

        live_max = (live[max_index] - artifact[max_index].reshape(-1, 1)).mean(0)
        live_min = (live[min_index] - artifact[min_index].reshape(-1, 1)).mean(0)

        live_max_interp = np.interp(wl, calib_wl, live_max)
        live_min_interp = np.interp(wl, calib_wl, live_min)


        plt.figure()
        plt.plot(wl, live_max_interp, label="max")
        plt.plot(wl, live_min_interp, label="min")
        plt.legend()
        plt.grid()
        plt.xlabel("wavelength[nm]")
        plt.ylabel("reflectance[-]")
        plt.savefig(os.path.join(output_path, "live", input_date + "_" + n + ".png"))
        plt.clf()

        df_dict = {}
        df_dict["wavelength"] = wl
        df_dict["max"] = live_max_interp 
        df_dict["min"] = live_min_interp

        df = pd.DataFrame(df_dict)

        df.to_csv(os.path.join(output_path, "live", input_date + "_" + n + ".csv"), index=None)


def preprocess_phantom_muscle(input_date):
    
    # setting
    input_path = os.path.join("data", "raw", input_date, "muscle")
    output_path = os.path.join("data", "processed", input_date, "muscle")

    if not os.path.isdir(input_path):
        raise Exception("There is no raw data with id: {}".format(input_date))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(os.path.join(output_path, "muscle")):
        os.mkdir(os.path.join(output_path, "muscle"))


    bg_path = os.path.join(input_path, "background")
    calib_wl_path = os.path.join("data", "raw", input_date, "calib_wl.csv")
    phantom_path = os.path.join(input_path, "phantom")
    
    if not os.path.isdir(bg_path):
        raise Exception("background.csv does not exist!")

    if not os.path.isfile(calib_wl_path):
        raise Exception("calib_wl.csv does not exist!")

    if not os.path.isdir(phantom_path):
        raise Exception("Folder phantom_muscle does not exist!")

    wl = [i for i in range(650, 1001, 10)]

    # load background
    bg = []
    for b in glob(os.path.join(bg_path, "*.csv")):
        bg.append(np.loadtxt(b, delimiter=","))

    # shape: [拍攝張數, 長, 寬]
    bg = np.asarray(bg)

    # load calibration wavelength
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')

    phantom_list = glob(os.path.join(phantom_path, "*"))
    chiken = {pid: i for i, pid in enumerate("chiken")}
    phantom_list.sort(key=lambda x: chiken[x])

    data = []
    for pl in phantom_list:
        _data = []
        for p in glob(os.path.join(pl, "*")):
            # 這裡每一筆data是2維的影像！
            _data.append(np.loadtxt(p, delimiter=","))
        data.append(_data)

    # shape: [phantom數, 拍攝數, 長, 寬]
    data = np.asarray(data)

    # shape: [phantom數, 長=1600, 寬=200]
    data_sub_bg = data.mean(1) - bg.mean(0)

    # 取中間波長
    data_middle = data_sub_bg[:, data_sub_bg.shape[1]//2, :]

    data = []
    for pid, dm in enumerate(data_middle):
        # 先找peaks
        pks = find_peaks_cwt(dm, np.arange(1, 30))
        assert len(pks) == 5, "peak number incorrect!"

        _data = []        
        for p in pks:
   
            peak_value = data_middle[p]
            left_found = False
            right_found = False
            left_idx = 0
            right_idx = 0
            for i in range(30):

                if not left_found and data_middle[p-i] < peak_value/2:
                    left_found = True
                    left_idx = i
                if not right_found and data_middle[p+i] < peak_value/2:
                    right_found = True
                    right_idx = i
                if left_found and right_found:
                    break

            # 其中一個仿體的其中一個SDS
            _data.append(data_sub_bg[pid, :, left_idx:right_idx].mean(1))

        data.append(_data)

    # shape: [phantom數, SDS數, 波長數]
    data = np.asarray(data)



    data_interp = []
    for d in data:
        data_interp.append(np.interp(wl, calib_wl, d[1]))

    # shape: [phantom數, SDS數, 校正後波長數(36)]
    data_interp = np.asarray(data_interp)

    # # plot and
    # # save
    # df_dict = {}
    # df_dict['wavelength'] = wl
    # for p, d in zip(phantom_list, data_interp):
    #     df_dict[p[-5]] = d
    #     plt.plot(wl, d, label=p[-5])
    # plt.grid()
    # plt.legend()
    # plt.xlabel("wavelength[nm]")
    # plt.ylabel("reflectance[-]")
    # plt.savefig(os.path.join(output_path, "phantom", input_date + ".png"))
    # plt.clf()

    # df = pd.DataFrame(df_dict)
    # df.to_csv(os.path.join(output_path, "phantom", input_date + ".csv"), index=None)

    # return data_interp

def preprocess_live_muscle(input_date):
    pass


def calibrate(input_date, sim_path="CHIKEN/sim_chik.csv", p_index="chik"):
    calib = Calibrator()
    p_index = list(p_index)

    input_path = os.path.join("data", "processed", input_date)
    live_path = os.path.join(input_path, "live")
    phantom_path = os.path.join(input_path, "phantom", input_date + ".csv")

    output_path = os.path.join("data", "calibrated", input_date)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # fit
    phantom = pd.read_csv(phantom_path)
    sim_phantom = pd.read_csv(sim_path)

    phantom = phantom[p_index].values.T
    sim_phantom = sim_phantom[p_index].values.T

    calib.fit(phantom, sim_phantom)

    live_list = glob(os.path.join(live_path, input_date + "*.csv"))
    for l in live_list:
        idx = l.split(input_date)[-1]
        df = pd.read_csv(l)

        df["max"] = calib.calibrate(df["max"].values)[0]
        df["min"] = calib.calibrate(df["min"].values)[0]

        df.to_csv(os.path.join(output_path, input_date + idx), index=None)


if __name__ == "__main__":
    import sys
    date = sys.argv[1]
    print(date)
    preprocess_phantom(date)
    preprocess_live(date)
    calibrate(date)




