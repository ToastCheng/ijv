import numpy as np 
import pandas as pd 


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
        for idx, m in enumerate(measured):
            assert m.shape == self.a.shape, "input shape does not match!"
            measured[idx] = self.a * m + self.b

        return measured

if __name__ == "__main__":

    c = Calibrator()

    x1 = np.expand_dims(np.arange(0, 5, 1), 1)
    y1 = x1 * 3 + 1
    y1[4] = x1[4] * 3 + 2

    x2 = np.expand_dims(np.arange(10, 15, 1), 1)
    y2 = x2 * 7 + 2
    y2[4] = x2[4] * 2 + 1

    x = np.concatenate([x1, x2], 1)
    y = np.concatenate([y1, y2], 1)

    print(x.shape)

    c.fit(x, y)

    # exec(open('utils/calibration.py').read())
    # exec(open('calibration.py').read())