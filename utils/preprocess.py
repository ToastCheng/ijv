import os
import pandas as pd 
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks_cwt
from scipy.optimize import fmin
from PyEMD import EMD


from kde import smooth

wl = [i for i in range(660, 921, 10)]


class SegmentCalibrator:
    def __init__(self):
        self.a = []
        self.b = []

    def fit(self, measured, simulated, cross_valid=False, plot_path=None, least_square=True):
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
        if cross_valid:
            worst = None
            for one_out in range(-1, num_p):
                index = [i for i in range(num_p) if i != one_out]
                _measured = measured[index]
                _simulated = simulated[index]
                a = []
                b = []

                _r_square = []
                for idx, (x, y) in enumerate(zip(_measured.T, _simulated.T)):
                    
                    if least_square:
                        aa, bb = np.polyfit(x, y, 1)
                    else:
                        rmsp = lambda i: (((i[0] * x + i[1] - y)/y)**2).mean()
                        aa, bb = fmin(rmsp, x0=[0, 0])

                    a.append(aa)
                    b.append(bb)

                    y_fit = x * a[idx] + b[idx]
                    residual = ((y-y_fit)**2).sum()
                    SS_total = ((y.mean()-y)**2).sum()
                    _r_square.append(1 - residual/SS_total)

                print("leave: %d, r_square: %.2f" % (one_out, np.mean(_r_square)))
                if np.mean(_r_square) > r_square_max:
                    worst = one_out
                    self.a = np.asarray(a)
                    self.b = np.asarray(b)
                    r_square_max = np.mean(_r_square)
                    r_square = _r_square.copy()

            if plot_path:
                for idx, (x, y) in enumerate(zip(measured.T, simulated.T)):
                    plt.scatter(x, y)
                    plot_x = np.arange(x[0], x[-1])
                    plt.plot(plot_x, plot_x * self.a[idx] + self.b[idx])
                    plt.ylim(0, y.max()*1.1)
                    plt.savefig(os.path.join(plot_path, "calibrate_{}.png".format(wl[idx])))
                    plt.clf()
            if worst:
                print("finally leave: {}".format(worst))
            else:
                print("keep all phantom")
        else:
            r_square = []
            a = []
            b = []
            bound = []
            for i, (x, y) in enumerate(zip(measured.T, simulated.T)):

                # 分段fit!
                aa = []
                bb = []
                bound_ = []
                for p in range(num_p-1):

                    # polyfit
                    if least_square:
                        result = np.polyfit(x[p:p+2], y[p:p+2], 1)
                    else:
                    # fmin
                        rmsp = lambda i: (((i[0] * x[p:p+2] + i[1] - y[p:p+2])/y[p:p+2]**2)).mean()
                        result = fmin(rmsp, x0=[0, 0])

                    aa += [result[0]]
                    bb += [result[1]]
                    bound_ += [x[p+1]]

                    y_fit = x[p:p+2] * result[0] + result[1]
                    residual = ((y[p:p+2]-y_fit)**2).sum()
                    SS_total = ((y[p:p+2].mean()-y[p:p+2])**2).sum()
                    r_square += [1 - residual/SS_total]


                    if plot_path:
                        plt.scatter(x[p:p+2], y[p:p+2])
                        plot_x = np.arange(x[p], x[p+1])
                        plt.plot(plot_x, plot_x * result[0] + result[1])

                a += [aa]
                b += [bb]
                bound += [bound_]

                

                if plot_path:
                    plt.ylim(0, y.max()*1.1)
                    plt.savefig(os.path.join(plot_path, "calibrate_{}.png".format(wl[i])))
                    plt.clf()

            # a.shape, b.shape, bound.shape
            # [波長, 區間數]
            self.a = np.asarray(a)
            self.b = np.asarray(b)
            self.bound = np.asarray(bound)

            # 調整校正係數
            bw1 = 2.2
            bw2 = 0.6
            for i in range(self.a.shape[1]):
                self.a[:, i] = smooth(self.a[:, i], bw1)
                self.b[:, i] = smooth(self.b[:, i], bw2)



        return self.a, self.b, r_square

    def calibrate(self, measured):
        # measured.shape
        # [活體數, 波長]
        measured = np.asarray(measured)
        if len(measured.shape) == 1:
            measured = np.expand_dims(measured, 0)



        calibrated = []
        for idx, m in enumerate(measured):
            # m.shape 
            # [波長]
            if not m.shape == self.a[:, 0].shape:
                print("measured shape: ", m.shape)
                print("calibrate shape: ", self.a.shape)
                raise Exception("input shape does not match!")

            calibrated_ = []
            for w_idx, (mm, bound) in enumerate(zip(m, self.bound)):
                section = 0
                for b in bound[:-1]:
                    if mm > b:
                        section += 1
                    else:
                        break

                calibrated_ += [self.a[w_idx, section] * mm + self.b[w_idx, section]]
                # calibrated_ += [self.a[:, section] * mm + self.b[:, section]]

            calibrated += [calibrated_]

        return np.asarray(calibrated)



class Calibrator:
    def __init__(self):
        self.a = []
        self.b = []

    def fit(self, measured, simulated, cross_valid=True, plot_path=None, least_square=True):
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
        if cross_valid:
            worst = None
            for one_out in range(-1, num_p):
                index = [i for i in range(num_p) if i != one_out]
                _measured = measured[index]
                _simulated = simulated[index]
                a = []
                b = []

                _r_square = []
                for idx, (x, y) in enumerate(zip(_measured.T, _simulated.T)):
                    
                    if least_square:
                        aa, bb = np.polyfit(x, y, 1)
                    else:
                        rmsp = lambda i: (((i[0] * x + i[1] - y)/y)**2).mean()
                        aa, bb = fmin(rmsp, x0=[0, 0])

                    a.append(aa)
                    b.append(bb)

                    y_fit = x * a[idx] + b[idx]
                    residual = ((y-y_fit)**2).sum()
                    SS_total = ((y.mean()-y)**2).sum()
                    _r_square.append(1 - residual/SS_total)

                print("leave: %d, r_square: %.2f" % (one_out, np.mean(_r_square)))
                if np.mean(_r_square) > r_square_max:
                    worst = one_out
                    self.a = np.asarray(a)
                    self.b = np.asarray(b)
                    r_square_max = np.mean(_r_square)
                    r_square = _r_square.copy()

            if plot_path:
                for idx, (x, y) in enumerate(zip(measured.T, simulated.T)):
                    plt.scatter(x, y)
                    plot_x = np.arange(x[0], x[-1])
                    plt.plot(plot_x, plot_x * self.a[idx] + self.b[idx])
                    plt.ylim(0, y.max()*1.1)
                    plt.savefig(os.path.join(plot_path, "calibrate_{}.png".format(wl[idx])))
                    plt.clf()
            if worst:
                print("finally leave: {}".format(worst))
            else:
                print("keep all phantom")
        else:
            r_square = []
            a = []
            b = []
            for i, (x, y) in enumerate(zip(measured.T, simulated.T)):


                # polyfit
                if least_square:
                    result = np.polyfit(x, y, 1)
                else:
                # fmin
                    rmsp = lambda i: (((i[0] * x + i[1] - y)/y)**2).mean()
                    result = fmin(rmsp, x0=[0, 0])


                a += [result[0]]
                b += [result[1]]

                y_fit = x * result[0] + result[1]
                residual = ((y-y_fit)**2).sum()
                SS_total = ((y.mean()-y)**2).sum()
                r_square += [1 - residual/SS_total]

                if plot_path:
                    plt.scatter(x, y)
                    plot_x = np.arange(x[0], x[-1])
                    plt.plot(plot_x, plot_x * result[0] + result[1])
                    plt.ylim(0, y.max()*1.1)
                    plt.savefig(os.path.join(plot_path, "calibrate_{}.png".format(wl[i])))
                    plt.clf()

            self.a = np.asarray(a)
            self.b = np.asarray(b)


        return self.a, self.b, r_square

    def calibrate(self, measured):

        measured = np.asarray(measured)
        if len(measured.shape) == 1:
            measured = np.expand_dims(measured, 0)

        calibrated = []
        for idx, m in enumerate(measured):
            if not m.shape == self.a.shape:
                print("measured shape: ", m.shape)
                print("calibrate shape: ", self.a.shape)
                raise Exception("input shape does not match!")
            calibrated += [self.a * m + self.b]


        return np.asarray(calibrated)


def get_hwfm(spec, output_path, num_sds=5):
    # spec: 2D spectrum image
    
    plt.figure()
    spec_mean = spec.sum(1)
    plt.plot(spec_mean)
    peak_idx = find_peaks_cwt(spec_mean, range(1, 10))
    if len(peak_idx) < num_sds:
        raise Exception("find {} peaks, but expect more than {}!".format(len(peak_idx), num_sds))

    peak_idx = list(peak_idx)
    peak_idx.sort(key=lambda x: spec_mean[x-3:x+3].max(), reverse=True)
    # peak_idx.sort(key=lambda x: spec_mean[x], reverse=True)
    peak_idx = peak_idx[:num_sds]

    # 確認抓到的peak是正確的
    for i, pp in enumerate(peak_idx):
        plt.scatter(pp, spec_mean[pp])
        plt.text(x=pp, y=spec_mean[pp], s=str(i+1))
        
    def fwhm(peak, signal):
        m = signal[peak]
        l = r = peak
        while signal[l] > m/2 and l > 0:
            l -= 1
            if peak - l > 10:
                l = np.argmin(signal[l:peak]) + l
                break
        while signal[r] > m/2 and r < len(signal)-1:
            r += 1
            if r - peak > 10:
                r = np.argmin(signal[peak:r]) + peak
                break
        return l, r   
    
    phantom = []
    for i, pp in enumerate(peak_idx):
        l, r = fwhm(pp, spec_mean - spec_mean.min())
        plt.plot([l, l], [0, spec_mean[l]], color="C{}".format(i))
        plt.plot([r, r], [0, spec_mean[r]], color="C{}".format(i))
        
        phantom += [spec[l:r].mean(0)]
    phantom = np.asarray(phantom)


    plt.xlabel("y-direction")
    plt.ylabel("intensity")
    plt.grid()

    plt.savefig(output_path)
    plt.clf()
    
    return phantom


def preprocess_phantom(input_date):
    
    # setting
    input_path = os.path.join("data", "raw", input_date, "IJV")
    output_path = os.path.join("data", "processed", input_date)

    if not os.path.isdir(input_path):
        raise Exception("There is no raw data with id: {}".format(input_date))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, "IJV")

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
    plt.figure(figsize=(12, 6))
    plt.xticks(wl)

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
    input_path = os.path.join("data", "raw", input_date, "IJV")
    output_path = os.path.join("data", "processed", input_date, "IJV")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(os.path.join(output_path, "live")):
        os.mkdir(os.path.join(output_path, "live"))

    bg_path = os.path.join(input_path, "background.csv")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    live_path = os.path.join(input_path, "live")

    bg = np.loadtxt(bg_path, delimiter=",")
    calib_wl = np.loadtxt(calib_wl_path, delimiter=',')


    live_list = glob(os.path.join(live_path, "live_*_*.csv"))
    live_list.sort(key=lambda x: (x.split('_')[1], x.split('_')[2].strip(".csv")))
    live_idx = [i.split("/")[-1].strip("live_").strip(".csv") for i in live_list]

        
    for l, n in zip(live_list, live_idx):
        try:
            # [time, wavelength(1600)]
            live = np.loadtxt(l)
        except ValueError:
            live = np.loadtxt(l, delimiter=",")

        live -= bg.mean(0)


        live_crop = live[:, (calib_wl > 600) & (calib_wl < 950)]
        live_crop = live_crop.mean(1)
        # live_crop = 1 - live_crop/65535

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

        # plt.figure()
        # plt.plot(live_max, label="max")
        # plt.plot(live_min, label="min")
        # plt.legend()
        # plt.grid()
        # plt.xlabel("wavelength[nm]")
        # plt.ylabel("reflectance[-]")
        # plt.savefig(os.path.join(output_path, "live", input_date + "_test_" + n + ".png"))
        # plt.clf()

        live_max_interp = np.interp(wl, calib_wl, live_max)
        live_min_interp = np.interp(wl, calib_wl, live_min)


        plt.figure(figsize=(12, 6))
        plt.xticks(wl)
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
    output_path = os.path.join("data", "processed", input_date)

    if not os.path.isdir(input_path):
        raise Exception("There is no raw data with id: {}".format(input_date))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(os.path.join(output_path, "muscle")):
        os.mkdir(os.path.join(output_path, "muscle"))

    if not os.path.isdir(os.path.join(output_path, "muscle", "phantom")):
        os.mkdir(os.path.join(output_path, "muscle", "phantom"))

    output_path = os.path.join(output_path, "muscle", "phantom")


    bg_path = os.path.join(input_path, "background_*")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    
    phantom_path = os.path.join(input_path, "phantom")
    live_path = os.path.join(input_path, "live")

    if len(glob(bg_path)) < 1:
        raise Exception("background file does not exist!")

    if not os.path.isfile(calib_wl_path):
        raise Exception("{} does not exist!".format(calib_wl_path))

    if not os.path.isdir(phantom_path):
        raise Exception("{} does not exist!".format(phantom_path))


    # load calibration wavelength
    try:
        calib_wl = np.loadtxt(calib_wl_path)
    except ValueError:
        calib_wl = np.loadtxt(calib_wl_path, delimiter=',')


    # load background
    bg = []
    for b in glob(bg_path):
        try:
            bg += [np.loadtxt(b)]
        except ValueError:
            bg += [np.loadtxt(b, delimiter=',')]


    # shape: [拍攝張數, 長, 寬]
    bg = np.asarray(bg)
    bg = bg.mean(0)
    # phantom
    p_id = "chik"

    phantom = []
    for pp in p_id:
        p_list = glob(os.path.join(phantom_path, "phantom_{}_*.*".format(pp)))
        phantom_ = []
        for p in p_list:
            try:
                phantom_ += [np.loadtxt(p)]
            except ValueError:
                phantom_ += [np.loadtxt(p, delimiter=',')]
            
        phantom_ = np.asarray(phantom_).mean(0) - bg
        phantom_ = get_hwfm(phantom_, os.path.join(output_path, "hwfm_{}.png".format(pp)))
        phantom += [phantom_]
    
    # [仿體數, SDS數, 波長數(1600)] 
    phantom = np.asarray(phantom)



    phantom_interp = []
    for p in phantom:
        phantom_ = []
        for pp in p:
            phantom_ += [np.interp(wl, calib_wl, pp)]
        phantom_interp += [np.asarray(phantom_)]

    # [仿體數, SDS數, 波長數(篩選後的)]
    phantom_interp = np.asarray(phantom_interp)

    for i, p in zip(p_id, phantom_interp):

        pd.DataFrame({
            "wavelength": wl,
            "0": p[0],
            "1": p[1],
            "2": p[2],
            "3": p[3],
            "4": p[4],
            }).to_csv(os.path.join(output_path, "phantom_{}.csv".format(i)), index=None)


        plt.figure(figsize=(12, 6))

        for idx, pp in enumerate(p):
            plt.plot(wl, pp, label="SDS: {}".format(idx))

        plt.xticks(wl)
        plt.xlabel("wavelength [nm]")
        plt.ylabel("reflectance [-]")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, "phantom_{}.png".format(i)), index=None)
        plt.clf()

    return phantom_interp


def preprocess_live_muscle(input_date):
    
    # setting
    input_path = os.path.join("data", "raw", input_date, "muscle")
    output_path = os.path.join("data", "processed", input_date)

    if not os.path.isdir(input_path):
        raise Exception("There is no raw data with id: {}".format(input_date))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if not os.path.isdir(os.path.join(output_path, "muscle")):
        os.mkdir(os.path.join(output_path, "muscle"))

    if not os.path.isdir(os.path.join(output_path, "muscle", "live")):
        os.mkdir(os.path.join(output_path, "muscle", "live"))

    output_path = os.path.join(output_path, "muscle", "live")


    bg_path = os.path.join(input_path, "background_*")
    calib_wl_path = os.path.join(input_path, "calib_wl.csv")
    
    live_path = os.path.join(input_path, "live")

    if len(glob(bg_path)) < 1:
        raise Exception("background file does not exist!")

    if not os.path.isfile(calib_wl_path):
        raise Exception("{} does not exist!".format(calib_wl_path))

    if not os.path.isdir(live_path):
        raise Exception("{} does not exist!".format(live_path))

    # load calibration wavelength
    try:
        calib_wl = np.loadtxt(calib_wl_path)
    except ValueError:
        calib_wl = np.loadtxt(calib_wl_path, delimiter=',')


    # load background
    bg = []
    for b in glob(bg_path):
        try:
            bg += [np.loadtxt(b)]
        except ValueError:
            bg += [np.loadtxt(b, delimiter=',')]

    # shape: [拍攝張數, 長, 寬]
    bg = np.asarray(bg)
    bg = bg.mean(0)

    # live
    live_list = glob(os.path.join(live_path, "*"))
    live_list.sort(key=lambda x: (x.split("live_")[-1].split('_')[0], x.split("live_")[-1].split('_')[1]))
    if len(live_list) % 5 != 0:
        raise Exception("live folder seems conatain other files")

    live = []
    for i in range(int(len(live_list)/5)):
        live_ = []
        for l in live_list[i*5:(i+1)*5]:
            try: 
                live_ += [np.loadtxt(l)]
            except ValueError:
                live_ += [np.loadtxt(l, delimiter=',')]

        live_ = np.asarray(live_).mean(0) - bg
        live_ = get_hwfm(live_, os.path.join(
            output_path, 
            "hwfm_{}.png".format(live_list[i*5].split("live_")[-1].split("_X")[0]))
        )
        live += [live_]
        
    # [活體數, SDS數, 波長數(1600)
    np.asarray(live)

    live_interp = []
    for l in live:
        live_ = []
        for ll in l:
            live_ += [np.interp(wl, calib_wl, ll)]
        live_interp += [np.asarray(live_)]
    
    # [活體數, SDS數, 波長數(篩選後的)]
    live_interp = np.asarray(live_interp)



    for i, l in enumerate(live_interp):

        i = live_list[i*5].split("live_")[-1].split("_X")[0]

        pd.DataFrame({
            "wavelength": wl,
            "0": l[0],
            "1": l[1],
            "2": l[2],
            "3": l[3],
            "4": l[4],
            }).to_csv(os.path.join(output_path, "live_{}.csv".format(i)), index=None)

        plt.figure(figsize=(12, 6))

        for idx, ll in enumerate(l):
            plt.plot(wl, ll, label="SDS: {}".format(idx))
        
        plt.xticks(wl)
        plt.xlabel("wavelength [nm]")
        plt.ylabel("reflectance [-]")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, "live_{}.png".format(i)), index=None)
        plt.clf()

    return live_interp


def calibrate_ijv(input_date, sim_path="CHIKEN/sim_20190525_24mm.csv", p_index="chik"):
    calib = SegmentCalibrator()
    p_index = list(p_index)

    input_path = os.path.join("data", "processed", input_date, "IJV")
    live_path = os.path.join(input_path, "live")
    phantom_path = os.path.join(input_path, "phantom", input_date + ".csv")

    output_path = os.path.join("data", "calibrated", input_date)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
    output_path = os.path.join("data", "calibrated", input_date, "IJV")

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # fit ijv
    phantom = pd.read_csv(phantom_path)
    sim_phantom = pd.read_csv(sim_path)

    phantom = phantom[p_index].values.T
    sim_phantom = sim_phantom[p_index].values.T

    calib.fit(phantom, sim_phantom, cross_valid=False, plot_path=output_path, least_square=True)

    live_list = glob(os.path.join(live_path, input_date + "*.csv"))
    for l in live_list:
        idx = l.split(input_date)[-1]
        df = pd.read_csv(l)

        df["max"] = calib.calibrate(df["max"].values)[0]
        df["min"] = calib.calibrate(df["min"].values)[0]

        plt.figure(figsize=(12, 6))
        plt.plot(wl, df["max"], label="max")
        plt.plot(wl, df["min"], label="min")
        plt.xlabel("wavelength [nm]")
        plt.ylabel("reflectance [-]")
        plt.xticks(wl)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, input_date + idx.strip(".csv") + ".png"))

        df.to_csv(os.path.join(output_path, input_date + idx), index=None)

def calibrate_muscle(input_date, sim_path="CHIKEN/phantom_muscle.npy", p_index="chik"):

    p_index = list(p_index)


    input_path = os.path.join("data", "processed", input_date, "muscle")
    live_path = os.path.join(input_path, "live")
    phantom_path = os.path.join(input_path, "phantom")

    output_path = os.path.join("data", "calibrated", input_date)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, "muscle")

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    phantom = []
    for pid in p_index:
        phantom_ = pd.read_csv(os.path.join(phantom_path, "phantom_{}.csv".format(pid)))

        # [SDS數, 波長數]
        phantom_ = phantom_.iloc[:, 1:].values.T
        phantom += [phantom_]
    # [仿體數, SDS數, 波長數]
    phantom = np.asarray(phantom)

    # [仿體數, SDS數, 波長數]
    phantom_sim = np.load(sim_path)

    assert phantom.shape == phantom_sim.shape, "shape should be the same!"

    calib_list = [Calibrator() for _ in range(phantom_sim.shape[1])]

    for i in range(len(calib_list)):
        calib_list[i].fit(phantom[:, i, :], phantom_sim[:, i, :], cross_valid=True, plot_path=output_path)

    live_list = glob(os.path.join(live_path, "live_*.csv"))
    for l in live_list:
        df = pd.read_csv(l)
        idx = l.split("live_")[-1].strip(".csv")

        # [SDS, wl] 
        df = df.iloc[:, 1:].values.T


        num_sds = df.shape[0]
        num_wl = df.shape[1]

        live_calib = []
        for s in range(num_sds):
            live_calib += [calib_list[s].calibrate(df[s])[0]]
        
        # [SDS, wl]
        live_calib = np.asarray(live_calib)

        plt.figure(figsize=(12, 6))
        for i, lc in enumerate(live_calib):
            plt.plot(wl, lc, label=str(i))
        
        plt.xlabel("wavelength [nm]")
        plt.ylabel("reflectance [-]")
        plt.xticks(wl)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, input_date + idx + ".png"))

        df = pd.DataFrame({
            "wavelength": wl,
            "0": live_calib[0],
            "1": live_calib[1],
            "2": live_calib[2],
            "3": live_calib[3],
            "4": live_calib[4] 
            })

        df.to_csv(os.path.join(output_path, input_date + idx + ".csv"), index=None)



    # calib.fit(phantom.reshape(phantom.shape[0], -1), phantom_sim.reshape(phantom.shape[0], -1))

    # live_list = glob(os.path.join(live_path, "live_*.csv"))
    # for l in live_list:
    #     idx = l.split("live_")[-1].strip(".csv")
    #     df = pd.read_csv(l)

    #     df = df.iloc[:, 1:].values.T
    #     num_sds = df.shape[0]
    #     num_wl = df.shape[1]

    #     df = df.reshape(-1)

    #     df = calib.calibrate(df)
    #     df = df.reshape(num_sds, num_wl)

    #     plt.figure(figsize=(12, 6))
    #     for i, d in enumerate(df):
    #         plt.plot(wl, d, label=str(i))
        
    #     plt.xlabel("wavelength [nm]")
    #     plt.ylabel("reflectance [-]")
    #     plt.xticks(wl)
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(os.path.join(output_path, input_date + idx + ".png"))

    #     df = pd.DataFrame(df)
    #     df["wavelength"] = wl 
    #     df = df[["wavelength"] + [str(i) for i in range(num_sds)]]
    #     df.to_csv(os.path.join(output_path, input_date + idx + ".csv"), index=None)










if __name__ == "__main__":
    import sys
    date = sys.argv[1]
    print(date)

    # print("process ijv..")
    # preprocess_phantom(date)
    # preprocess_live(date)

    # print("calibrate ijv..")
    # calibrate_ijv(date)

    # print("ps_live_muscle(date)

    print("calibrate muscle..")
    calibrate_muscle(date)


