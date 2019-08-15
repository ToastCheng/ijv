from utils import load_mch
from utils import load_mc2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
import numpy as np 

from tqdm import tqdm


dr = load_mc2("output/quicktest_path_high_mus/mcx_output/quicktest_path_high_mus_700.mc2", (150, 150, 200))
dr_zm = zoom(dr, (0.5, 0.5, 0.5))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
for i in tqdm(range(50)):
    for j in range(25, 45):
        for k in range(20):
            if dr_zm[i,j,19-k] < dr_zm.max()/8:
                continue
            ax.scatter(i, j, k, alpha=(dr_zm[i,j,19-k]-dr_zm.min())/(dr_zm.max()-dr_zm.min()), color="#ff2222")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# for i in tqdm(range(100)):
#     for j in range(65, 85):
#         for k in range(40):
#             ax.scatter(i, j, k, alpha=dr[i,j,k]/dr.max(), color="#eeee55")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()
