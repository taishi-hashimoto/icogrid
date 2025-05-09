# %%
from icogrid import make_icogrid, ndiv_from_angle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as _
from tqdm.auto import tqdm

angular_separation =1
n = ndiv_from_angle(angular_separation, degrees=True)
print("n =", n)
for _ in tqdm(range(1)):
    points = make_icogrid(n)
print(points)

# %% Compare results with old library.
import numpy as np
from anglelib.icosphere import Icosphere

for _ in tqdm(range(1)):
    ref = Icosphere.from_angular_separation(angular_separation, degrees=True).vertices


# Check points in 3-d plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", s=angular_separation)
ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c="k", s=10*angular_separation, alpha=0.5)
fig.tight_layout()
plt.show()

assert np.allclose(np.sort(ref, axis=0),  np.sort(points, axis=0)), "The points do not match the reference points."
# %%
