import numpy as np


def hex_lattice(a: float, b: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Make a hexagonal grid and return the flatten array"""
    nx1 = int(np.ceil(2 / np.sqrt(3) * a / pitch) + 1)
    nx2 = int(np.ceil(2 / np.sqrt(3) * b / pitch) + 1)
    q, r = np.mgrid[-nx1 : nx1 + 1, -nx2 : nx2 + 1]
    x1_hex = pitch * (q + 0.5 * r)
    x2_hex = pitch * (np.sqrt(3) / 2.0) * r

    return x1_hex.ravel(), x2_hex.ravel()


def rect_lattice(
    a: float, b: float, dx1: float, dx2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Make a rectangular grid and return the flatten array"""
    nx1 = int(np.ceil(a / dx1))
    nx2 = int(np.ceil(b / dx2))
    q, r = np.mgrid[-nx1 : nx1 + 1, -nx2 : nx2 + 1]
    x1_rect = q * dx1
    x2_rect = r * dx2
    return x1_rect.ravel(), x2_rect.ravel()


def line(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    """Make a straight line along x3 axis and return the flatten array"""
    nx3 = int(np.ceil(end - start) / spacing)
    x3 = np.linspace(start, end, nx3)
    print(x3.shape)
    x2 = np.zeros_like(x3)
    x1 = np.zeros_like(x3)
    return np.stack((x1, x2, x3), axis=1)


def arc(
    center: np.ndarray,
    radius: float,
    angle_start: float,
    angle_end: float,
    angle_step: float,
) -> np.ndarray:
    center = np.asarray(center)
    ang_len = angle_end - angle_start
    na = int(np.ceil(ang_len / angle_step))
    ang = np.linspace(angle_start, angle_end, na)
    x1 = center[0] + radius * np.cos(ang)
    x2 = np.zeros_like(x1)
    x3 = center[1] + radius * np.sin(ang)
    return np.stack((x1, x2, x3), axis=1)


def circle(
    center: np.ndarray, radius: float, pitch: float, lattice: str = "hex"
) -> np.ndarray:
    center = np.asarray(center)
    if lattice == "hex":
        x1, x2 = hex_lattice(radius, radius, pitch)
    elif lattice == "rect":
        x1, x2 = rect_lattice(radius, radius, pitch, pitch)

    mask = x1**2 + x2**2 <= radius**2
    x3 = np.zeros_like(x1[mask])
    return np.stack((x1[mask] + center[0], x2[mask] + center[1], x3), axis=1)


def polygon(vertex: np.ndarray, pitch: float, lattice: str = "hex") -> np.ndarray:
    from matplotlib.path import Path

    vertex = np.asarray(vertex)
    center = vertex.mean(axis=0)
    vertex_cent = vertex - center  # el centrado vive aquí, escondido
    radius = np.abs(vertex_cent).max()  # caja envolvente, ya resuelta
    if lattice == "hex":
        x1, x2 = hex_lattice(radius, radius, pitch)
    elif lattice == "rect":
        x1, x2 = rect_lattice(radius, radius, pitch, pitch)
    points = np.stack([x1.ravel(), x2.ravel()], axis=1)
    mask = Path(vertex_cent).contains_points(points).reshape(x1.shape)

    x3 = np.zeros_like(x1[mask])
    return np.stack((x1[mask], x2[mask], x3), axis=1)


def rectangle(
    center: np.ndarray,
    width: float,
    height: float,
    dx1: float,
    dx2: float,
    plane: str = "xy",
) -> np.ndarray:
    from matplotlib.path import Path

    center = np.asarray(center)
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    center_0 = vertex.mean(axis=0)
    vertex_cent = vertex - center_0
    x1, x2 = rect_lattice(width, height, dx1, dx2)
    points = np.stack([x1.ravel(), x2.ravel()], axis=1)
    mask = Path(vertex_cent).contains_points(points).reshape(x1.shape)
    x3 = np.zeros_like(x1[mask])
    if plane == "xy":
        points = np.stack((x1[mask] + center[0], x2[mask] + center[1], x3), axis=1)
    elif plane == "xz":
        points = np.stack((x1[mask] + center[0], x3, x2[mask] + center[1]), axis=1)
    elif plane == "yz":
        points = np.stack((x3, x1[mask] + center[0], x2[mask] + center[1]), axis=1)
    return points


def cylinder(radius: float, height: float, spacing: float) -> np.ndarray:
    # pendiente, §5
    pass


def cone() -> np.ndarray:
    # pendiente, §5
    pass


def sphere(radius: float, half_angle: float, n_z: int) -> np.ndarray:
    # pendiente, §5
    pass
