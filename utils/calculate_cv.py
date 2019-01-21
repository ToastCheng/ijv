from copy import copy
import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt

results_path = glob("test_cv_*/result/result.pkl")
results_path.sort(key=lambda x: int(x[8]))

def cal_cv(data, axis=0):
    # 輸出 CV值 (in percentage)
    return data.std(axis)/data.mean(axis) * 100

spec = []
for p in results_path:
    spec.append(np.load(p))
spec = np.asarray(spec)



for i in range(5):
    plt.figure(figsize=(14, 6))
    plt.plot(cv[91, i, :])
    plt.xlabel("wavelength")
    plt.ylabel("CV")
    plt.xticks([i for i in range(35)], [str(i) for i in range(650, 1001, 10)])
    plt.title("SDS: %.1fmm" % sds[i])
    plt.tight_layout()
    plt.savefig("cv_scv90_%d.png" % i)


# Before smooth
for i in range(5):
    print(cv[91, i, :].mean())


spec_smooth = copy(spec)
for i in range(6):
    for j in range(101):
        for k in range(5):
            spec_smooth[i, j, k, :] = kde.fit(spec[i, j, k, :], 3)


cv_smooth = cal_cv(spec_smooth)

# after smooth
for i in range(5):
    print(cv_smooth[91, i, :].mean())

for i in range(5):
    plt.figure(figsize=(14, 6))
    plt.plot(cv_smooth[91, i, :])
    plt.xlabel("wavelength")
    plt.ylabel("CV")
    plt.xticks([i for i in range(35)], [str(i) for i in range(650, 1001, 10)])
    plt.title("SDS: %.1fmm" % sds[i])
    plt.tight_layout()
    plt.savefig("cv_scv90_%d.png" % i)

plt.figure(figsize=(16, 10))
for i in range(6):
    plt.plot(spec[i, 91, 0, :], label="%d" % i)
plt.legend()
plt.show()


plt.imshow(cv_smooth.mean(2), aspect="auto")
plt.xlabel("SDS")
plt.ylabel("ScvO2")
plt.colorbar()
# plt.show()
plt.savefig("cv_hm_scv.png")
plt.clf()
plt.imshow(cv_smooth.mean(0).T, aspect="auto")
plt.xlabel("SDS")
plt.ylabel("wavelength")
plt.colorbar()
plt.savefig("cv_hm_wl.png")

# plt.show()

