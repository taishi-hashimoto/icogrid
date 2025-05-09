"Icosahedral grid generator."
from math import sqrt, atan2
import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit, prange
import jax
import jax.numpy as jnp


def make_icosahedron():
    "Generate an icosahedron that is aligned such that two vertices are on the Z axis."
    r3 = sqrt(3)
    r5 = sqrt(5)
    icosahedron = [
        (1, r3, (-3-r5)/2),
        (-2, 0, (-3-r5)/2),
        (1, -r3, (-3-r5)/2),
        (-(1+r5)/2, -(1+r5)*r3/2, (1-r5)/2),
        (1+r5, 0, (1-r5)/2),
        (-(1+r5)/2, (1+r5)*r3/2, (1-r5)/2),
        ((1+r5)/2, (1+r5)*r3/2, (r5-1)/2),
        (-1-r5, 0, (r5-1)/2),
        ((1+r5)/2, -(1+r5)*r3/2, (r5-1)/2),
        (-1, -r3, (3+r5)/2),
        (2, 0, (3+r5)/2),
        (-1, r3, (3+r5)/2),
    ]
    # Rotate to align two vertices on Z axis.
    icosahedron = Rotation.from_rotvec(
        -atan2(icosahedron[10][0], icosahedron[10][2]) * np.array([0, 1, 0])
    ).apply(icosahedron)
    # Make sure these vertices have exact zeros in X and Y coordinates.
    icosahedron[1, (0, 1)] = 0.
    icosahedron[10, (0, 1)] = 0.
    return np.array(icosahedron)


def make_triangles():
    "Generate triangles list on the icosahedron."
    triangles = [
        (0, 1, 5),
        (0, 5, 6),
        (0, 6, 4),
        (0, 4, 2),
        (0, 2, 1),
        (1, 7, 5),
        (1, 3, 7),
        (1, 2, 3),
        (2, 8, 3),
        (2, 4, 8),
        (7, 11, 5),
        (5, 11, 6),
        (6, 11, 10),
        (6, 10, 4),
        (9, 10, 11),
        (7, 9, 11),
        (10, 8, 4),
        (3, 9, 7),
        (3, 8, 9),
        (8, 10, 9),
    ]
    return np.array(triangles)


@njit(cache=True)
def divide_triangle(vertices: np.ndarray, triangle: np.ndarray, n: int):
    """Divide a triangle by n, and return all vertices and indices.
    
    Parameters
    ==========
    vertices : ndarray of float
        Vertices of the icosahedron [12, 3].
    triangle : ndarray of int
        Single triangle on the icosahedron to be divided [3].
    n : int
        Number of subdivisions.

    Returns
    =======
    triangles, indices: tuple
        triangles: list of vertices of the triangles.
        indices: list of tuples, indices of each vertex of triangles.
        The lengths of these are the same and is equal to n**2.
    """
    ia: int
    ib: int
    ic: int
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    ia, ib, ic = triangle

    # a is the top, b is the left bottom, and c is the right bottom vertex.
    a, b, c = vertices[ia], vertices[ib], vertices[ic]
    
    # Lazy way of determining the number of items `m` in the array...
    # m = 0
    # for i in range(n):
    #     for j in range(i+1):
    #         m += 1
    #         if j != i:
    #             m += 1
    m = n**2

    u = (b - a) / n
    v = (c - b) / n
    index = 0
    indices = np.empty((m, 3, 2, 3), dtype=np.int64)
    triangles = np.empty((m, 3, 3), dtype=np.float64)
    for i in range(n):
        for j in range(i+1):
            # p is on a -> b
            iabi = (ia, ib, i)
            ibcj = (ib, ic, j)
            iabi1 = (ia, ib, i+1)
            ibcj1 = (ib, ic, j+1)
            ip = (iabi, ibcj)
            p = a + i*u + j*v
            # q is on a -> b and end point.
            iq = (iabi1, ibcj)
            q = p + u
            # r is on right of q
            ir = iabi1, ibcj1
            r = q + v
            indices[index, 0, :, :] = np.array(ip)
            indices[index, 1, :, :] = np.array(iq)
            indices[index, 2, :, :] = np.array(ir)
            triangles[index, 0, :] = p
            triangles[index, 1, :] = q
            triangles[index, 2, :] = r
            index += 1
            # s is right of p
            js = iabi, ibcj1
            s = p + v
            if j != i:
                indices[index, 0, :, :] = np.array(ip)
                indices[index, 1, :, :] = np.array(ir)
                indices[index, 2, :, :] = np.array(js)
                triangles[index, 0, :] = p
                triangles[index, 1, :] = r
                triangles[index, 2, :] = s
                index += 1
    assert index == m
    return triangles, indices


@njit(cache=True)
def _divide_triangle(icosahedron, triangle, n):
    "Call divide_triangle and return the points and indices."
    tri, ind = divide_triangle(icosahedron, triangle, n)
    
    m = 3 * n**2
    
    indices = np.empty((m, 6), dtype=np.int64)
    points = np.empty((m, 3), dtype=np.float64)
    for j, i, t in zip(range(m), ind.reshape(-1, 2, 3), tri.reshape(-1, 3)):
        (ia, ib, ii), (jb, jc, jj) = i
        if ii == 0: # a
            assert jj == 0
            ib = ia
            jb = -1
            jc = -1
        elif ii == n and jj == 0: # b
            ia = ib
            ii = 0
            jb = -1
            jc = -1
        elif ii == n and jj == n: # c
            ia = jc
            ib = jc
            ii = 0
            jb = -1
            jc = -1
            jj = 0
        elif ii == n: # on bc
            ia = jb
            ib = jc
            ii = jj
            jb = -1
            jc = -1
            jj = 0
        elif ii == jj: # on ac
            ib = jc
            jb = -1
            jc = -1
            jj = 0

        if jj == 0: # on ab
            jb = -1
            jc = -1

        if ia > ib:
            ia, ib = ib, ia
            ii = n - ii

        if jb > jc:
            jb, jc = jc, jb
            jj = n - jj
        
        if jb != -1 and ia > jb:
            ia, ib, ii, jb, jc, jj = jb, jc, jj, ia, ib, ii

        # i = (ia, ib, ii), (jb, jc, jj)
        indices[j, :] = np.array([ia, ib, ii, jb, jc, jj])
        points[j, :] = t
    return points, indices


@njit(cache=True, parallel=True)
def _make_icogrid(icosahedron, triangles, n):
    "icogrid body."
    # points = {}
    # for triangle in triangles:
    nt = len(triangles)
    m = 3*n**2
    points = np.empty((m*nt, 3), dtype=np.float64)
    indices = np.empty((m*nt, 6), dtype=np.int64)
    for it in prange(nt):
        ib = it * m
        ie = ib + m
        _points, _indices = _divide_triangle(icosahedron, triangles[it], n)
        points[ib:ie, :] = _points
        indices[ib:ie, :] = _indices
    return points, indices


# jitted version of np.unique
jit_unique = jax.jit(jnp.unique, static_argnames=["return_index", "size", "axis"])


def make_icogrid(n: int):
    """Generate an icosahedral grid.
    
    The resultant vectors are normalized. 

    Parameters
    ----------
    n : int
        Number of subdivisions.

    Returns
    -------
    points: ndarray of float
        Points on the icosahedral grid [N, 3].
        The result is normalized.
    """

    icosahedron = make_icosahedron()
    triangles = make_triangles()
    _points, _indices = _make_icogrid(icosahedron, triangles, n)
    _, idx = jit_unique(_indices, axis=0, return_index=True, size=10 * n**2 + 2)
    # _, idx = np.unique(_indices, axis=0, return_index=True)
    points = _points[idx]
    assert len(points) == 10 * n**2 + 2
    points = np.array(points) / np.linalg.norm(points, axis=-1, keepdims=True)
    return points


def ndiv_from_angle(
    angular_separation: float,
    degrees: bool = False
) -> int:
    """Calculate the number of subdivisions from the angular separation.
    
    Parameters
    ----------
    angular_separation : float
        Angular separation in degrees.

    Returns
    -------
    n : int
        Number of subdivisions.
    """
    if not degrees:
        angular_separation = np.rad2deg(angular_separation)
    n = 1
    while True:
        if 69 / n < angular_separation:
            break
        else:
            n += 1
    return n


# precompile
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # デフォルトではWARNINGなので注意

start = time.time()
logger.info("Compiling Numba functions...")
_ = _make_icogrid(make_icosahedron(), make_triangles(), 1)
elapsed = time.time() - start
logger.info(f"Numba compilation completed in {elapsed:.2f} seconds.")