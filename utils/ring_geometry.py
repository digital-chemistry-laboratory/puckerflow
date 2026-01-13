"""Low-level geometry helpers used across PuckerFlow.

These utilities implement the numerical conversions between Cartesian
coordinates, Fourier coefficients, and Cremer-Pople amplitudes, along with
handy planar geometry helpers that operate directly on torch tensors.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch


def fix_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace numerically zero tensors with explicit zeros.

    Args:
        x: Input tensor that may contain near-zero elements.

    Returns:
        Tensor with the same shape as ``x`` where values within the tolerance
        are snapped exactly to zero.
    """

    if torch.allclose(x, torch.zeros_like(x), rtol=1e-6, atol=1e-8):
        return torch.zeros_like(x)
    return x


def translate(coordinates: torch.Tensor) -> torch.Tensor:
    """Center coordinates at their geometric mean.

    Args:
        coordinates: Cartesian positions shaped ``(N, 3)``.

    Returns:
        Positions shifted so that their centroid lies at the origin.
    """

    return coordinates - coordinates.mean(dim=0, keepdim=False)


def gram_schmidt(
    R1: torch.Tensor, R2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orthonormalize ``R1``/``R2`` using Gram-Schmidt.

    Args:
        R1: First spanning vector of the target plane.
        R2: Second spanning vector of the target plane.

    Returns:
        Tuple ``(R1_hat, R2_hat)`` holding unit-length orthogonal vectors.

    Raises:
        ValueError: If ``R1`` is near zero or ``R1``/``R2`` are colinear.
    """

    R1_norm = torch.norm(R1)
    if R1_norm < 1e-8:
        raise ValueError("Cannot normalize zero-length vector R1")
    R1_hat = R1 / R1_norm

    projection = torch.dot(R2, R1_hat) * R1_hat
    R2_orthogonal = R2 - projection
    R2_norm = torch.norm(R2_orthogonal)
    if R2_norm < 1e-8:
        raise ValueError("R1 and R2 are linearly dependent")
    R2_hat = R2_orthogonal / R2_norm
    return R1_hat, R2_hat


def get_mean_plane(coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute orthonormal axes spanning the ring's mean plane.

    Args:
        coordinates: Cartesian ring coordinates shaped ``(N, 3)``.

    Returns:
        Tuple ``(R1, R2)`` that forms an orthonormal basis of the mean plane.
    """

    N = coordinates.shape[0]
    angles = (
        2
        * math.pi
        * torch.arange(N, device=coordinates.device, dtype=coordinates.dtype)
        / N
    )
    cos_terms = torch.cos(angles).unsqueeze(0)
    sin_terms = torch.sin(angles).unsqueeze(0)

    R1 = fix_zero(torch.flatten(torch.matmul(cos_terms, coordinates)))
    R2 = fix_zero(torch.flatten(torch.matmul(sin_terms, coordinates)))

    return gram_schmidt(R1, R2)


def get_normal(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the unit normal vector of the mean plane.

    Args:
        coordinates: Cartesian ring coordinates shaped ``(N, 3)``.

    Returns:
        Unit-length normal vector orthogonal to the mean plane.

    Raises:
        ValueError: If the cross product magnitude is near zero.
    """

    R1, R2 = get_mean_plane(coordinates)
    cross_product = torch.cross(R1, R2, dim=0)
    norm = torch.norm(cross_product)
    if norm < 1e-8:
        raise ValueError("Degenerate normal vector (zero magnitude)")
    return cross_product / norm


def displacements(coordinates: torch.Tensor) -> torch.Tensor:
    """Project coordinates onto the mean-plane normal.

    Args:
        coordinates: Cartesian positions shaped ``(N, 3)``.

    Returns:
        One-dimensional tensor containing the signed distances along the
        plane normal.
    """

    normal = get_normal(coordinates)
    return torch.matmul(coordinates, normal)


def z_to_ab(z: torch.Tensor) -> torch.Tensor:
    """Convert mean-plane displacements to Fourier ``a``/``b`` coefficients.

    Args:
        z: Tensor of mean-plane displacements shaped ``(N,)``.

    Returns:
        Tensor containing stacked ``a``/``b`` Fourier coefficients.
    """

    N = z.shape[0]
    spectrum = torch.fft.rfft(z, norm="ortho")

    ab: List[torch.Tensor] = []
    num_pairs = max(0, math.floor((N - 3) / 2))
    for j in range(num_pairs):
        coeff = spectrum[2 + j]
        ab.append(math.sqrt(2.0) * torch.stack((coeff.real, coeff.imag)))

    ab_tensor = torch.cat(ab) if ab else torch.empty(0, device=z.device, dtype=z.dtype)

    if N % 2 == 0:
        ab_tensor = torch.cat((ab_tensor, spectrum[N // 2].real.unsqueeze(0)))

    return ab_tensor


def ab_to_cp(ab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Fourier coefficients to Cremer-Pople amplitudes and phases.

    Args:
        ab: Tensor storing Fourier coefficients in ``[a0, b0, a1, b1, ...]`` order.

    Returns:
        Tuple ``(cp_amp, cp_ang)`` containing amplitudes and phases.
    """

    num_ab = ab.shape[0]
    num_angles = math.floor(num_ab / 2)
    num_amps = math.ceil(num_ab / 2)

    cp_amp = torch.empty(num_amps, device=ab.device)
    cp_ang = torch.empty(num_angles, device=ab.device)

    for i in range(num_angles):
        a_val = ab[2 * i]
        b_val = ab[2 * i + 1]
        cp_amp[i] = torch.sqrt(a_val * a_val + b_val * b_val)
        cp_ang[i] = torch.atan2(b_val, a_val)

    if num_amps > num_angles:
        cp_amp[-1] = ab[-1]

    return cp_amp, cp_ang


def ab_to_z(ab: torch.Tensor) -> torch.Tensor:
    """Reconstruct ``z`` displacements from Fourier coefficients.

    Args:
        ab: Tensor of Fourier coefficients.

    Returns:
        Tensor of mean-plane displacements corresponding to ``ab``.
    """

    num_ab = ab.shape[0]
    num_pairs = num_ab // 2

    real_parts = ab[0 : 2 * num_pairs : 2] / math.sqrt(2.0)
    imag_parts = ab[1 : 2 * num_pairs : 2] / math.sqrt(2.0)
    complex_part = torch.complex(real_parts, imag_parts)

    if num_ab % 2 == 1:
        complex_part = torch.cat(
            (complex_part, ab[-1].to(complex_part.dtype).unsqueeze(0))
        )

    zeros_part = torch.zeros(2, dtype=complex_part.dtype, device=ab.device)
    spectrum = torch.cat((zeros_part, complex_part))
    return torch.fft.irfft(spectrum, norm="ortho")


def get_ring_pucker_coords(
    coordinates: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Cremer-Pople amplitudes and phases from coordinates.

    Args:
        coordinates: Cartesian ring coordinates shaped ``(N, 3)``.

    Returns:
        Tuple ``(cp_amp, cp_ang)`` describing the Cremer-Pople parameters.
    """

    z = displacements(translate(coordinates))
    ab = z_to_ab(z)
    return ab_to_cp(ab)


def tilt_coordinate_system(
    normal: torch.Tensor, R1: torch.Tensor, R2: torch.Tensor
) -> torch.Tensor:
    """Create a rotation matrix from a mean plane and its normal.

    Args:
        normal: Unit vector perpendicular to the plane.
        R1: First in-plane axis.
        R2: Second in-plane axis.

    Returns:
        Rotation matrix where columns correspond to ``(R1, R2, normal)``.
    """

    rotation_matrix = torch.stack((R1, R2, normal), dim=1)
    return rotation_matrix


def fourier_position(pos: torch.Tensor) -> torch.Tensor:
    """Transform 3D coordinates into the Fourier (mean plane) basis.

    Args:
        pos: Cartesian coordinates shaped ``(N, 3)``.

    Returns:
        Coordinates expressed in the orthonormal Fourier basis.
    """

    centered = translate(pos)
    R1, R2 = get_mean_plane(centered)
    normal = get_normal(centered)
    rotation = tilt_coordinate_system(normal, R1, R2)
    return centered @ rotation


def is_convex_polygon(points: torch.Tensor) -> torch.Tensor:
    """Check whether a 2D polygon is convex.

    Args:
        points: Tensor of 2D vertices shaped ``(N, 2)``.

    Returns:
        Boolean tensor indicating if the polygon is convex.
    """

    points_closed = torch.cat([points, points[:2]], dim=0)
    v1 = points_closed[1:-1] - points_closed[:-2]
    v2 = points_closed[2:] - points_closed[1:-1]

    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    non_zero_cross = cross[cross != 0]
    signs = torch.sign(non_zero_cross)
    return torch.all(signs == signs[0]) if signs.numel() > 0 else torch.tensor(True)


def polygon_angles(vertices: torch.Tensor) -> torch.Tensor:
    """Calculate internal angles of a 2D polygon in degrees.

    Args:
        vertices: Tensor of 2D vertices shaped ``(N, 2)``.

    Returns:
        Tensor containing one interior angle per vertex (in degrees).
    """

    prev_vertices = torch.roll(vertices, shifts=-1, dims=0)
    next_vertices = torch.roll(vertices, shifts=1, dims=0)

    vec1 = prev_vertices - vertices
    vec2 = next_vertices - vertices

    vec1_norm = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2_norm = vec2 / vec2.norm(dim=1, keepdim=True)

    dot_product = torch.sum(vec1_norm * vec2_norm, dim=1).clamp(-1, 1)
    cross_product = (
        vec1_norm[:, 0] * vec2_norm[:, 1] - vec1_norm[:, 1] * vec2_norm[:, 0]
    )

    angles_rad = torch.atan2(cross_product, dot_product)
    angles_deg = angles_rad * 180 / math.pi
    return torch.where(angles_deg > 0, angles_deg, 360 + angles_deg)


def calculate_convexity(pos: torch.Tensor) -> Tuple[bool, List[float]]:
    """Return convexity flag and internal angles for a 2D projection.

    Args:
        pos: Cartesian coordinates shaped ``(N, 3)``.

    Returns:
        Tuple ``(is_convex, internal_angles)`` describing the polygon.
    """

    convex = is_convex_polygon(pos[:, :2]).item()
    internal_angles = polygon_angles(pos[:, :2]).tolist()
    if convex and any(angle > 180 for angle in internal_angles):
        print("Convexity code errs!")
    return convex, internal_angles


def calculate_z_diffs(pos: torch.Tensor) -> List[float]:
    """Calculate absolute differences in ``z`` displacement between neighbors.

    Args:
        pos: Cartesian coordinates shaped ``(N, 3)``.

    Returns:
        List with ``N`` entries describing consecutive absolute differences.
    """

    z = displacements(pos)
    N = z.shape[0]
    return [abs(z[i] - z[(i + 1) % N]).item() for i in range(N)]


def pairwise_norm(p: np.ndarray) -> List[float]:
    """Calculate the L2 norm of adjacent pairs in a 1D array.

    Args:
        p: One-dimensional numpy array containing values to group.

    Returns:
        List of pairwise norms, with the final element preserved when the
        input length is odd.
    """

    N = p.size
    norms: List[float] = []
    for i in range(0, N - 1, 2):
        norms.append(float(np.linalg.norm(p[i : i + 2])))
    if N % 2 != 0:
        norms.append(float(p[-1]))
    return norms
