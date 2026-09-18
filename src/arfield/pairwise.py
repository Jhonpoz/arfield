"""Pairwise geometry between two clouds of points.

The only module that turns two sets of coordinates into the distance and
direction of every pair, and the only place an ``(M, N, 3)`` array is built
from scratch. Everything above it -- the Green's kernel, the influence
matrices, the field evaluation -- works from what comes out of here and
never touches coordinates again.

No physics, no wavenumber, no medium: this is arithmetic on points.
"""

import numpy as np

__all__ = ["separation", "separation_distance"]


def _difference(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Difference vector from every source to every target.

    The only place in the package where coordinates are subtracted, and the
    only place where their dtype is decided: integer input is converted to
    ``float64`` here so that both public functions behave the same. Change
    that policy here and nowhere else.

    Parameters
    ----------
    targets : array_like, shape (M, 3)
        Points ``x_T``, in meters.
    sources : array_like, shape (N, 3)
        Points ``y_S``, in meters.

    Returns
    -------
    ndarray, shape (M, N, 3)
        ``x_T - y_S``, float64, from source to target. Newly allocated;
        callers may modify it in place.
    """
    targets = np.asarray(targets, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)
    return targets[:, None, :] - sources[None, :, :]


def _distance(R_TSj: np.ndarray) -> np.ndarray:
    """Euclidean norm of a difference array along its last axis.

    Written as a self-contraction rather than ``np.linalg.norm`` so that no
    second ``(M, N, 3)`` array is built; see the notes in `separation`.

    Parameters
    ----------
    R_TSj : ndarray, shape (M, N, 3)
        Difference vectors, float64.

    Returns
    -------
    ndarray, shape (M, N)
        ``|R_TSj|``, float64.
    """
    return np.sqrt(np.einsum("mnj,mnj->mn", R_TSj, R_TSj))


def separation_distance(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Distance from every source to every target, without the direction.

    Same distances as `separation` returns first, bit for bit, for callers
    that need only ``r_TS`` -- `influence.compute_green_TS` among them.
    The ``(M, N, 3)`` difference array is still built, but it is dropped on
    return instead of being turned into unit vectors and handed back, which
    is the whole point of this function.

    This does not call `separation`, and `separation` does not call this:
    each needs its own difference array, and the one built here is not
    reused for anything.

    Parameters
    ----------
    targets : array_like, shape (M, 3)
        Points ``x_T`` where the field is evaluated, in meters.
    sources : array_like, shape (N, 3)
        Point-source positions ``y_S``, in meters.

    Returns
    -------
    ndarray, shape (M, N)
        ``|x_T - y_S|``, float64, in meters.
    """
    return _distance(_difference(targets, sources))


def separation(
    targets: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance and unit vector from every source to every target.

    Both outputs are indexed ``(target, source)`` in that order, which is
    the shape convention of every influence matrix in the package.

    The unit vector points **from the source to the target**. That
    direction is what makes `green_kernel.gradient_green` return the
    gradient with respect to the target coordinate, which is the one the
    boundary condition and the field evaluation both need. Reversing it
    flips the sign of every gradient in the package and of nothing else, so
    the error survives every check that looks at magnitudes.

    Parameters
    ----------
    targets : array_like, shape (M, 3)
        Points ``x_T`` where the field is evaluated, in meters.
    sources : array_like, shape (N, 3)
        Point-source positions ``y_S``, in meters.

    Returns
    -------
    r_TS : ndarray, shape (M, N)
        ``|x_T - y_S|``, float64, in meters.
    e_TSj : ndarray, shape (M, N, 3)
        ``(x_T - y_S) / r_TS``, float64, from source to target.

    Notes
    -----
    Integer input is converted to ``float64`` in `_difference`; both public
    functions of this module share that policy.

    Coincident points are not handled. ``r_TS`` comes back as zero and the
    corresponding unit vector as ``nan``, after a division warning. Nothing
    downstream checks for it, so the warning is the only signal that the
    source layer was placed on top of a target.

    Both lines after the subtraction are written to avoid a temporary the
    size of the difference array, and both are load-bearing. Using
    ``np.linalg.norm`` for the distances squares that array elementwise into
    a second ``(M, N, 3)`` before reducing it; contracting the array with
    itself accumulates straight into the ``(M, N)`` result and never
    materializes the product. Dividing out of place makes a third copy.

    The cost of that temporary is mostly time, not peak memory. Peak goes
    from 2.00 times the bytes returned to 1.25, measured at every size from
    200 to 2000 squared and stable across runs, but no caller in this
    package peaks here: each of them allocates more further down, so the
    package-level maximum does not move. What does move is the clock,
    because the temporary still has to be written and read: about 1.8 times
    faster here, and a fifth off the field evaluation calls end to end, on
    one machine and one core. The ratio of peaks is fixed by the code and
    travels; the speedup is not and does not.

    A test asserts the ratio of peaks rather than the timing, since it is
    the stable proxy for the same mechanism, and neither line looks
    load-bearing without it.

    The two spellings of the distance disagree in the last bits, up to two
    ulps, from the order in which three terms are summed. Neither is more
    accurate and neither guards against overflow: ``np.linalg.norm`` along
    an axis squares without rescaling too, and overflows on the same
    inputs. Reference data generated before this changed needs
    regenerating, and comparisons against it need a tolerance.
    """
    e_TSj = _difference(targets, sources)
    r_TS = _distance(e_TSj)
    e_TSj /= r_TS[:, :, None]
    return r_TS, e_TSj
