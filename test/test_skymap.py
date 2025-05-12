# %% 
import numpy as np
import matplotlib.pyplot as plt
from icogrid import Icogrid, plot_skymap


if __name__ == "__main__":
    # Angular evaluation grid.
    angular_separation = 1
    zeaz = Icogrid.from_angular_separation(angular_separation, degrees=True).to_direction(degrees=True)
    zeaz = zeaz[zeaz[:, 0] < 10]
    # zeaz = zeaz[zeaz[:, 1] < np.deg2rad(90)]
    ndir = len(zeaz)
    
    r = 80e3
    
    ze_g, az_g = zeaz.T
    
    data =  np.abs(az_g - 30) * np.abs(ze_g - 5)

    fig, ax = plt.subplots()
    tpc = plot_skymap(ax, data, ze=ze_g, az=az_g,  r=r, cmap='viridis', shading='flat', degrees=True)
    plt.colorbar(tpc, ax=ax, shrink=0.6, orientation='horizontal', label='Example value')

    # %%
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(6, 6))
    tpc = plot_skymap(
        ax, data, ze=ze_g, az=az_g,
        degrees=True,
        cmap='viridis',
        shading='flat',
    )

    # Other settings for polar axes.
    # ax.set_theta_zero_location('N')
    # ax.set_theta_direction(-1)
    ax.set_rlim(0, 10)

    plt.colorbar(tpc, ax=ax, shrink=0.6, orientation='horizontal', label='Example value')
    plt.title("Triangulated Data on Polar Plot")
    plt.show()

    # %% Class
    ico = Icogrid.from_angular_separation(angular_separation, degrees=True)
    ico.set_extent(radius=r, zemin=0, zemax=10, degrees=True)
    dirs = ico.to_direction(degrees=True)
    data = np.abs(dirs[:, 1] - 30) * np.abs(dirs[:, 0] - 5)
    
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(6, 6))
    tpc = ico.plot(
        data,
        ax=ax,
    )
    ax.set_rlim(0, 10)
    plt.colorbar(tpc, ax=ax, shrink=0.6, orientation='horizontal', label='Example value')
    plt.title("Triangulated Data on Polar Plot")
    plt.show()

# %%
