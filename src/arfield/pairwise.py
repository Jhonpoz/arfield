import numpy as np


def separation(
    targets: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance and unit vector from every source to every target.

    Parameters
    ----------
    targets : (M, 3) ndarray
        Points x_T where the field is evaluated.
    sources : (N, 3) ndarray
        Point-source positions y_S.

    Returns
    -------
    r_TS : (M, N) ndarray
        |x_T - y_S|.
    e_TSj : (M, N, 3) ndarray
        (x_T - y_S) / r_TS, from source to target.
    """
    e_TSj = targets[:, None, :] - sources[None, :, :]
    r_TS = np.linalg.norm(e_TSj, axis=-1)
    e_TSj /= r_TS[:, :, None]  # in place: no second (M, N, 3) array

    return r_TS, e_TSj
