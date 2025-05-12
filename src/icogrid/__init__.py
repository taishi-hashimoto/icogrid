"Quasi-equispaced angular grid based on the vertices of icosahedral subdivision."
import numpy as np
import matplotlib.pyplot as plt
from .basics import make_icogrid, ndiv_from_angle
from .skymap import plot_skymap, direction


__all__ = [
    "make_icogrid",
    "plot_skymap",
    "ndiv_from_angle",
    "Icogrid",
]


class Icogrid:
    """Quasi-equispaced angular grid based on the vertices of icosahedral
    subdivision."""

    def __init__(self, n: int):
        self.n = n
        self.average_separation = np.deg2rad(69. / n)
        self._vertices = make_icogrid(n)
        "All vertices of the Icogrid in Cartesian coordinates."
        self._directions = np.c_[direction(self._vertices)]
        "Directions of the vertices in spherical coordinates."
        self.azmin = None
        self.azmax = None
        self.zemin = None
        self.zemax = None
        self.radius = None
        self._valid_mask = None

    @staticmethod
    def from_angular_separation(
        separation: float,
        degrees: bool = False
    ) -> 'Icogrid':
        """Generate an Icogrid with a specified angular separation.

        Parameters
        ==========
        separation: float
            Required angular separation between a closest pair in radian.
            n is automatically determined such that '69 / n < separation'.
        degrees: bool
            If True, the separation is in degrees.
        """
        return Icogrid(ndiv_from_angle(separation, degrees=degrees))

    def set_extent(
        self,
        azmin: float = None,
        azmax: float = None,
        zemin: float = None,
        zemax: float = None,
        radius: float = None,
        degrees: bool = False,
    ):
        """
        Set the extent of the Icogrid.
        The result is saved in valid_mask.

        Parameters
        ==========
        azmin: float
            Minimum azimuthal angle in radians.
        azmax: float
            Maximum azimuthal angle in radians.
        zemin: float
            Minimum zenith angle in radians.
        zemax: float
            Maximum zenith angle in radians.
        radius: float
            Radius of the sphere.
        degrees: bool
            If True, the angles are in degrees.

        Returns
        =======
        self: Icogrid
            The Icogrid object with the specified extent.
        """
        if azmin is not None and degrees:
            azmin = np.deg2rad(azmin)
        if azmax is not None and degrees:
            azmax = np.deg2rad(azmax)
        if zemin is not None and degrees:
            zemin = np.deg2rad(zemin)
        if zemax is not None and degrees:
            zemax = np.deg2rad(zemax)
        self.azmin = azmin
        self.azmax = azmax
        self.zemin = zemin
        self.zemax = zemax
        self.radius = radius
        
        valid_mask = np.ones(len(self._vertices), dtype=bool)
        if self.azmin is not None:
            valid_mask &= self._directions[:, 1] >= self.azmin
        if self.azmax is not None:
            valid_mask &= self._directions[:, 1] <= self.azmax
        if self.zemin is not None:
            valid_mask &= self._directions[:, 0] >= self.zemin
        if self.zemax is not None:
            valid_mask &= self._directions[:, 0] <= self.zemax
        self._valid_mask = valid_mask
        return self

    @property
    def valid_mask(self):
        """Valid mask of the vertices set by set_extent()."""
        if self._valid_mask is None:
            return slice(None)
        else:
            return self._valid_mask

    @property
    def vertices(self):
        """Cartesian coordinates of the vertices.
        
        valid_mask is applied.
        """
        return self._vertices[self.valid_mask, :]

    def to_direction(self, degrees: bool = False) -> np.ndarray:
        """Return directions of vertices in radians.
        
        valid_mask is applied.
        
        Parameters
        ==========
        degrees: bool
            If True, the angles are in degrees.
        """
        values = self._directions[self.valid_mask, :]
        if degrees:
            values = np.rad2deg(values)
        return values

    def plot(
        self,
        data,
        ax=None, projection=None,
        fig_kw: dict = None,
        subplot_kw: dict = None,
        gridspec_kw: dict = None,
        **kwargs
    ):
        """
        Plot the data on the Icogrid.

        Parameters
        ==========
        data: array-like
            Data to be plotted.
        ax: Axes
            Axes to plot on. If None, a new figure and axes will be created.
        cax: Axes
            Axes for colorbar. If None, a new colorbar will be created.
        projection: str
            Projection type. If None, a polar plot will be created.
        zmax: float
            Maximum value for the color scale. If None, the maximum value of
            the data will be used.
        fig_kw: dict
            Keyword arguments for the figure.
        subplot_kw: dict
            Keyword arguments for the subplot.
        gridspec_kw: dict
            Keyword arguments for the gridspec.
        kwargs: dict
            Keyword arguments for tripcolor().
        """
        zeaz = self.to_direction(degrees=True)
        ze, az = zeaz.T

        if ax is None:
            if fig_kw is None:
                fig_kw = {}
            if subplot_kw is None:
                subplot_kw = {}
                if projection is not None:
                    subplot_kw['projection'] = projection
            _, ax = plt.subplots(
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw,
                **fig_kw)
        return plot_skymap(
            ax=ax,
            ze=ze, az=az,
            r=self.radius,
            data=data,
            projection=projection,
            **kwargs
        )

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Icogrid(n={self.n})"
