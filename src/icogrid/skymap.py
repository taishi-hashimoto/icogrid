"Module for plotting skymaps on icogrid."
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes, PolarTransform
from matplotlib.tri import LinearTriInterpolator, Triangulation
from matplotlib.axes import Axes


def radial(
    ze: NDArray[np.float64],
    az: NDArray[np.float64],
    r: float = None,
    degrees: bool = False
):
    """Convert zenith and azimuth angles to radial coordinates.

    This function assumes 
    
    Parameters
    ==========
    ze: ndarray of float
        Zenith angles in radians.
    az: ndarray of float
        Azimuth angles in radians.
    r: float
        Distance from the origin to the grid points.
    
    Returns
    =======
    x, y, z: ndarray of float
        Radial coordinates.
    """
    # convert angular grids to xy
    if degrees:
        az = np.deg2rad(az)
    if r is None:  # radiation patten mode
        if not degrees:
            ze = np.rad2deg(ze)
        x = np.cos(az) * ze
        y = np.sin(az) * ze
        return x, y
    else:
        if degrees:
            ze = np.deg2rad(ze)
        x = r * np.cos(az) * np.sin(ze)
        y = r * np.sin(az) * np.sin(ze)
        z = r * np.cos(ze)
        return x, y, z


def direction(xyz: NDArray[np.float64], degrees: bool = False):
    "Inverse of radial for 3-d case."
    ze: NDArray[np.float64] = np.arccos(xyz[..., 2] / np.linalg.norm(xyz, axis=-1))
    az: NDArray[np.float64] = np.arctan2(xyz[..., 1], xyz[..., 0])
    if degrees:
        ze = np.rad2deg(ze)
        az = np.rad2deg(az)
    return ze, az


def xygrid(x, y, r: float = None, da=None, dx=None, dy=None, nx=50, ny=50):
    """Create a grid of points in the x-y plane.

    Parameters
    ==========
    x, y: ndarray of float
        X and Y coordinates of the grid points.
    r: float
        Distance from the origin to the grid points.
    da: float
        Hint for angular separation between the points.
        Actual value could be slightly different as it is used to compute the
        number of subdivisions.
    dx, dy: float
        Hint for X and Y grid spacing.
        This will be used if da is not specified.
    nx, ny: int
        Number of grid points in X and Y.
        This will be used if none of da, dx, dy is specified.
        Default is 50.
    """
    if r is not None and da is not None:
        dx = dy = np.min(np.tan(da) * r)
    if dx is not None:
        nx = int((np.max(x) - np.min(x)) / dx) + 1
        # dx = (np.max(x) - np.min(x)) / nx
    if dy is not None:
        ny = int((np.max(y) - np.min(y)) / dy) + 1
        # dy = (np.max(y) - np.min(y)) / ny
    return np.meshgrid(
        np.linspace(np.min(x), np.max(x), nx),
        np.linspace(np.min(y), np.max(y), ny),
    )


def triang_skymap(
    ze, az, r=None,
    degrees: bool=False,
    polar: bool = False
):
    """Triangulate the specified ze, az grid in the x-y plane.
    
    Parameters
    ==========
    ze: ndarray of float
        Zenith angles in radians.
    az: ndarray of float
        Azimuth angles in radians.
    r: float
        Distance from the origin to the grid points.
    degrees: bool
        If True, ze and az are in degrees.
    """
    if polar or r is None:  # polar plot mode.
        if degrees:
            az = np.deg2rad(az)
        else:
            ze = np.rad2deg(ze)
        # theta, r -> x, y
        x, y = PolarTransform().transform(np.c_[az, ze]).T
        tri = Triangulation(x, y)
    else:
        if r is None:
            raise ValueError("r must be specified for 3D triangulation.")
        x, y, _ = radial(ze, az, r, degrees=degrees)
        tri = Triangulation(x, y)
    return tri


def triang_xygrid(
    ze, az, r=None,
    degrees: bool=False,
    da: float = None,
    dx: float = None, dy: float = None,
    nx: int = 50, ny: int = 50,
):
    """Triangulate the specified ze, az grid in the x-y plane.
    
    Parameters
    ==========
    ze: ndarray of float
        Zenith angles in radians.
    az: ndarray of float
        Azimuth angles in radians.
    r: float
        Distance from the origin to the grid points.
    degrees: bool
        If True, ze and az are in degrees.
    da: float
        Angular separation between the points.
    dx, dy: float
        X and Y grid spacing.
        This will be used if da is not specified.
    nx, ny: int
        Number of grid points in X and Y.
        This will be used if none of da, dx, dy is specified.
    """
    # convert angular grids to xy
    x, y, _ = radial(ze, az, r, degrees)
    x_g, y_g = xygrid(x, y, r, da, dx, dy, nx, ny)
    triang = Triangulation(x, y)
    return x_g, y_g, triang


def interp_xygrid(
    x_g, y_g, triang,  # From triang_xygrid
    values
):
    interp = LinearTriInterpolator(triang, values)
    return interp(x_g, y_g)


def plot_skymap(
    ax: Axes,
    data: np.ndarray,
    ze: np.ndarray = None,
    az: np.ndarray = None, 
    r: float = None,
    tri: Triangulation = None,
    degrees: bool = False,
    pcolor: bool = False,
    line: bool = False,
    **kwargs
):
    """
    Plot the data on the Icogrid.

    Parameters
    ==========
    ax: Axes
        Axes to plot on. If None, a new figure and axes will be created.
    ze: array-like
        Zenith angles in radians.
    az: array-like
        Azimuth angles in radians.
    data: array-like
        Data to be plotted.
    r: float
        Distance from the origin to the grid points.
    tri: Triangulation
        Triangulation object. If None, a new triangulation will be created.
    degrees: bool
        If True, ze and az are in degrees.
    line: bool
        If True, a line plot will be generated by tricontour.
    kwargs: dict
        Keyword arguments for the tripcolor plot.
    """
    polar = isinstance(ax, PolarAxes)
    if tri is None:
        assert len(ze) == len(az) == len(data), (
            f"ze{np.shape(ze)}, az{np.shape(az)}, and data{np.shape(data)} must have the same length."
        )
        tri = triang_skymap(ze, az, r, degrees, polar)
    else:
        assert len(data) == len(tri.x), (
            f"data{np.shape(data)} and triangulation{np.shape(tri.x)} must have the same length."
        )
    if pcolor:
        method = ax.tripcolor
    elif line:
        method = ax.tricontour
    else:
        method = ax.tricontourf
    if polar:
        # tri is already in polar coordinates, so it is canceled out by proj.inverted().
        tpc = method(
            tri, data,
            transform=PolarTransform().inverted() + ax.transData,
            **kwargs)
        ax.set_rlim(0)
    else:
        tpc = method(tri, data, **kwargs)
        ax.set_aspect("equal")
    ax.grid(True)

    return tpc
