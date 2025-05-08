# %%
from icogrid import make_icogrid, ndiv_from_angle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as _

angular_separation = 10
n = ndiv_from_angle(10, degrees=True)
print("n =", n)
points = make_icogrid(n)
print(points)

# Check points in 3-d plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], s=angular_separation)
fig.tight_layout()
plt.show()


# %% Compare results with old library.
import numpy as np
from anglelib.icosphere import Icosphere

ref = Icosphere.from_angular_separation(angular_separation, degrees=True).vertices

assert np.allclose(points, ref, atol=1e-5), "The points do not match the reference points."
# %%
