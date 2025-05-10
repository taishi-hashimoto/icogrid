import numpy as np
from .basics import make_icogrid, ndiv_from_angle

__all__ = [
    "make_icogrid",
    "ndiv_from_angle",
]


class Icogrid:
    """Quasi-equispaced angular grid based on the vertives of icosahedral
    subdivision."""

    def __init__(self, n: int):
        self.n = n
        self.vertices = make_icogrid(n)
        self.average_separation = np.deg2rad(69. / n)

    @staticmethod
    def from_angular_separation(separation: float, degrees: bool = False):
        """
        Generate 
        separation:
            Required angular separation between a closest pair in degree.
            n is automatically determined such that '69 / n < separation'.
        """
        return Icogrid(ndiv_from_angle(separation, degrees=degrees))

    def to_direction(self, degrees: bool = False):
        """Return directions of vertices in radians."""
        results = np.column_stack((
            np.arccos(self.vertices[:, 2] / np.linalg.norm(self.vertices, axis=-1)),
            np.arctan2(self.vertices[:, 1], self.vertices[:, 0])
        ))
        if degrees:
            results = np.rad2deg(results)
        return results

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self):
        return len(self.vertices)

    def __repr__(self):
        return f"Icogrid(n={self.n})"
