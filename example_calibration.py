from utils.calibration import Calibrator 
from utils.kde import smooth

import numpy as np 
import pandas as pd 


# read data
# measurement
pd.read_csv("data/20190118.csv")

phantom = df.iloc[:, 0:5]
live = df.iloc[:, 5:]

# simulation
sim = []
for p in "CHIK":
    path = "output/%s.pkl" % p
    sim.append(np.load(path)[p])

sim[0] = np.asarray(sim[0])
sim[1] = np.asarray(sim[1])
sim[2] = np.asarray(sim[2])
sim[3] = np.asarray(sim[3])

# smooth
bw = 7
live["cy1"] = smooth(live["chenyi_lie15_1"], band_width=bw)
live["cy2"] = smooth(live["chenyi_lie2"], band_width=bw)
live = live[["cy1", "cy2"]]

phantom["PC"] = smooth(phantom["PC"], band_width=bw)
phantom["PH"] = smooth(phantom["PH"], band_width=bw)
phantom["PI"] = smooth(phantom["PI"], band_width=bw)
phantom["PK"] = smooth(phantom["PK"], band_width=bw)

bw = 0.8
sim[0] = smooth(sim[0].reshape(-1), band_width=bw)
sim[1] = smooth(sim[1].reshape(-1), band_width=bw)
sim[2] = smooth(sim[2].reshape(-1), band_width=bw)
sim[3] = smooth(sim[3].reshape(-1), band_width=bw)

sim = pd.DataFrame({
    "wavelength": [i for i in range(650, 1001, 10)],
    "C": sim[0],
    "H": sim[1],
    "I": sim[2],
    "K": sim[3]
})

# interpolate
wl = [i for i in range(650, 1001, 10)]
p_wl = phantom["Wavelength (nm)"].tolist()

phantom_interp = []
live_interp = []

for p in ["PC", "PH", "PI", "PK"]:
    phantom_interp.append(
        np.interp(wl, p_wl, phantom[p].tolist())
    )
for l in ["cy1", "cy2"]:
    live_interp.append(
        np.interp(wl, p_wl, live[l])
    )

phantom_interp = np.asarray(phantom_interp)
live_interp = np.asarray(live_interp)

sim = sim.iloc[:, 1:].values


# calibration
calib = Calibrator()
a, b, rr = calib.fit(phantom_interp, sim.T)


