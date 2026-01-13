"""
Functions for calculating 3D ring pucker coordinates from
internal coordinates (bond lengths, bond angles, and puckering parameters).

Based on the method by Cremer and Pople, and its extensions.
"""

import math
from typing import List
import torch


def get_z(ringsize: int, q: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Convert puckering amplitudes to mean-centered out-of-plane coordinates.

    Args:
        ringsize: Total number of atoms in the ring.
        q: Tensor of puckering amplitudes ``q_m``. Shape is ``(N-3)/2`` for odd
            rings and ``(N/2)-1`` for even rings.
        ang: Tensor of puckering phases ``phi_m``. Shape is ``(N-3)/2`` for odd
            rings and ``(N/2)-2`` for even rings.

    Returns:
        Tensor of size ``(ringsize,)`` containing the Z displacement for each
        atom, shifted such that the mean is exactly zero.

    Raises:
        ValueError: If the supplied ``q``/``ang`` sizes do not match ``ringsize``.
    """

    if not ((q.numel() + ang.numel()) == (ringsize - 3) and 3 < ringsize <= 16):
        raise ValueError(
            "Inappropriate Input: Mismatch in puckering parameters and ringsize"
        )

    device = q.device
    dtype = q.dtype
    z = torch.zeros(ringsize, device=device, dtype=dtype)

    if ringsize % 2 == 1:  # odd rings
        upper = int((ringsize - 1) / 2) + 1
        for m in range(2, upper):
            coeff = q[m - 2]
            phase = ang[m - 2]
            angles = (
                phase
                + 2.0
                * math.pi
                * m
                * torch.arange(ringsize, device=device, dtype=dtype)
                / ringsize
            )
            z = z + coeff * torch.cos(angles)
        z = math.sqrt(2.0 / ringsize) * z
    else:  # even rings
        upper = int(ringsize / 2)
        for m in range(2, upper):
            coeff = q[m - 2]
            phase = ang[m - 2]
            angles = (
                phase
                + 2.0
                * math.pi
                * m
                * torch.arange(ringsize, device=device, dtype=dtype)
                / ringsize
            )
            z = z + coeff * torch.cos(angles)
        z = math.sqrt(2.0 / ringsize) * z

        j = torch.arange(ringsize, device=device, dtype=dtype)
        alternating = 1 - 2 * (j % 2)
        z = z + math.sqrt(1.0 / ringsize) * q[-1] * alternating

    return z - torch.mean(z)


def _update_bond_length(
    ringsize: int, Z: torch.Tensor, init_bondl: torch.Tensor
) -> torch.Tensor:
    """Compute 2D bond lengths from 3D lengths and puckering offsets.

    Args:
        ringsize: Total number of atoms in the ring.
        Z: Mean-centered Z displacements, shape ``(ringsize,)``.
        init_bondl: Original 3D bond lengths, shape ``(ringsize,)``.

    Returns:
        Tensor of updated 2D-projected bond lengths with shape ``(ringsize,)``.

    Raises:
        ValueError: If the Z displacement differences violate triangle
            inequality and lead to negative squared bond lengths.
    """
    N = ringsize
    Z_next = torch.roll(Z, shifts=-1, dims=0)
    dZ = Z_next - Z

    r_sq = torch.square(init_bondl) - torch.square(dZ)

    if torch.any(r_sq < -1e-8):
        raise ValueError("Invalid bond lengths after update (sqrt of negative number)")

    nr = torch.sqrt(torch.clamp(r_sq, min=0.0))
    return nr


def _update_beta(
    ringsize: int,
    Z: torch.Tensor,
    r: torch.Tensor,
    beta: torch.Tensor,
    clip: bool = True,
) -> torch.Tensor:
    """Recalculate bond angles after applying puckering displacements.

    Args:
        ringsize: Number of atoms in the ring.
        Z: Puckering Z displacements, shape ``(ringsize,)``.
        r: Original 3D bond lengths, shape ``(ringsize,)``.
        beta: Original 3D bond angles, shape ``(ringsize,)``.
        clip: If True, clamp cosine arguments to ``[-1, 1]`` instead of raising.

    Returns:
        Tensor of updated bond angles (shape ``(ringsize,)``) appropriate for
        the 2D projection.

    Raises:
        ValueError: When ``clip`` is False and the cosine arguments exceed the
            valid domain for ``torch.acos``.
    """
    N = ringsize

    # Get new 2D bond lengths
    r_new = _update_bond_length(ringsize, Z, r)

    # Vectorized version of the loop
    # i, (i+1)%N, (i+2)%N
    i = torch.arange(N, device=Z.device)
    i_p1 = (i + 1) % N
    i_p2 = (i + 2) % N

    # Z coordinates
    Z_i = Z[i]
    Z_ip1 = Z[i_p1]
    Z_ip2 = Z[i_p2]

    # Bond lengths
    r_i = r[i]
    r_ip1 = r[i_p1]
    r_new_i = r_new[i]
    r_new_ip1 = r_new[i_p1]

    # Initial bond angles
    beta_i = beta[i]

    # Law of Cosines components
    # (Z[i[2]] - Z[i[0]])**2 - (Z[i[1]] - Z[i[0]])**2 - (Z[i[2]] - Z[i[1]])**2
    tmp_a = (
        torch.square(Z_ip2 - Z_i)
        - torch.square(Z_ip1 - Z_i)
        - torch.square(Z_ip2 - Z_ip1)
    )

    # 2 * r[i[0]] * r[i[1]] * torch.cos(beta[i[0]])
    tmp_b = 2 * r_i * r_ip1 * torch.cos(beta_i)

    # 2 * r_new[i[0]] * r_new[i[1]]
    tmp_c = 2 * r_new_i * r_new_ip1
    tmp_c = torch.where(torch.abs(tmp_c) < 1e-8, torch.full_like(tmp_c, 1e-8), tmp_c)

    temp = (tmp_a + tmp_b) / tmp_c

    if clip:
        temp = torch.clamp(temp, -1.0, 1.0)
    else:
        if torch.any((temp < -1.0 - 1e-8) | (temp > 1.0 + 1e-8)):
            raise ValueError("Invalid bond angle calculation (acos domain error)")

    beta_new = torch.acos(temp)

    if not clip and torch.isnan(beta_new).any():
        raise ValueError("Invalid bond angle calculation (acos domain error)")

    return beta_new


def _rotation_matrix(phi: torch.Tensor) -> torch.Tensor:
    """Return a 2x2 counter-clockwise rotation matrix for ``phi`` radians.

    Args:
        phi: Rotation angle in radians.

    Returns:
        Tensor with shape ``(2, 2)`` representing the rotation matrix.
    """
    c, s = torch.cos(phi), torch.sin(phi)
    row1 = torch.stack((c, -s))
    row2 = torch.stack((s, c))
    return torch.stack((row1, row2))


def _seg_coord(segment: List[int], r: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Generate 2D coordinates for a contiguous ring segment.

    Args:
        segment: Atom indices belonging to the segment.
        r: 1D tensor containing ring-wide bond lengths.
        beta: 1D tensor containing ring-wide bond angles.

    Returns:
        Tensor of shape ``(len(segment), 2)`` containing the planar coordinates.
    """
    segsize = len(segment)
    if segsize == 0:
        return torch.empty(0, 2, device=r.device, dtype=r.dtype)

    # Get bond lengths and angles for just this segment
    segment_r = r[segment]
    segment_beta = beta[segment]

    # Use torch functions instead of math for tensor operations
    alpha = [segment_beta[0]]
    gamma: List[torch.Tensor] = []

    # Use float tensors from the start
    coordinate = [
        torch.zeros(2, device=r.device, dtype=r.dtype),
        torch.stack((segment_r[0], torch.zeros((), device=r.device, dtype=r.dtype))),
    ]
    slength = [segment_r[0]]

    if segsize >= 2:
        for index in range(segsize - 1):
            angle = torch.pi - alpha[index]
            if index == 0:
                x = slength[-1] + segment_r[index + 1] * torch.cos(angle)
                y = segment_r[index + 1] * torch.sin(angle)
                coord = torch.stack((x, y))
                coordinate.append(coord)
            else:
                x = slength[-1] + segment_r[index + 1] * torch.cos(angle)
                y = segment_r[index + 1] * torch.sin(angle)
                rota = _rotation_matrix(gamma[index - 1])
                coord = torch.matmul(rota, torch.stack((x, y)))
                coordinate.append(coord)

            slength.append(torch.norm(coordinate[-1]))

            # Avoid division by zero
            slength_last = slength[-1]
            if slength_last == 0:
                slength_last = torch.tensor(1e-8, device=r.device, dtype=r.dtype)

            # Clamp arcsin input to avoid domain errors
            asin_input = torch.sin(alpha[-1]) * slength[index] / slength_last
            part_angle = torch.arcsin(torch.clamp(asin_input, -1.0, 1.0))

            if slength[index] > slength[index + 1] and slength[index] > r[index + 1]:
                part_angle = torch.pi - part_angle

            alpha.append(segment_beta[index + 1] - part_angle)
            gamma.append(torch.atan2(coord[1], coord[0]))

    return torch.stack(coordinate, dim=0)


def _ring_partition(
    ringsize: int,
    z: torch.Tensor,
    r: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Partition the ring and assemble 3D coordinates from segments.

    Args:
        ringsize: Number of atoms in the ring.
        z: Z displacements for each atom.
        r: Updated bond lengths.
        beta: Updated bond angles.

    Returns:
        Tensor of shape ``(ringsize, 3)`` containing centered coordinates.

    Raises:
        Exception: If ``ringsize`` falls outside the supported range.
    """

    location = list(range(ringsize))

    # Define segments based on ring size
    if 5 <= ringsize <= 7:
        s1, s2 = 2, -2
    elif 8 <= ringsize <= 10:
        s1, s2 = 3, -3
    elif 11 <= ringsize <= 13:
        s1, s2 = 4, -4
    elif 14 <= ringsize <= 16:
        s1, s2 = 5, -5
    elif ringsize == 4:
        s1, s2 = 1, -1
    else:
        raise Exception(f"Unproper ring size ({ringsize})")

    segment1 = location[0:s1]
    segment2 = location[s1:s2]
    segment3 = location[s2:]

    segcoord_1 = _seg_coord(segment1, r, beta)
    segcoord_2 = _seg_coord(segment2, r, beta)
    segcoord_3 = _seg_coord(segment3, r, beta)

    OPsq = torch.dot(segcoord_1[-1], segcoord_1[-1])
    PQsq = torch.dot(segcoord_2[-1], segcoord_2[-1])
    OQsq = torch.dot(segcoord_3[-1], segcoord_3[-1])

    phi1 = torch.atan2(segcoord_1[-1][1], segcoord_1[-1][0]) % math.pi
    phi2 = torch.atan2(segcoord_2[-1][1], segcoord_2[-1][0]) % math.pi
    phi3 = torch.atan2(segcoord_3[-1][1], segcoord_3[-1][0]) % math.pi

    xq = (OPsq + OQsq - PQsq) / (2 * torch.sqrt(OPsq))
    yq = torch.sqrt(PQsq - (torch.sqrt(OPsq) - xq) ** 2)

    phiseg1 = torch.atan2(yq, xq)
    phiseg2 = (math.pi - torch.atan2(yq, torch.sqrt(OPsq) - xq)) % math.pi
    phiseg3 = phiseg1 + math.pi

    sigma1 = -phi1
    sigma2 = phiseg2 - phi2
    sigma3 = phiseg3 - phi3

    Rsigma1 = _rotation_matrix(sigma1)
    Rsigma2 = _rotation_matrix(sigma2)
    Rsigma3 = _rotation_matrix(sigma3)

    coordinate_1 = [torch.zeros(2, dtype=z.dtype, device=z.device)]
    seg1_size = segcoord_1.shape[0]

    for i in range(1, seg1_size - 1):
        coordinate_1.append(torch.matmul(Rsigma1, segcoord_1[i]))
    sqrt_OP = torch.sqrt(torch.clamp(OPsq, min=0.0))
    coordinate_1.append(
        torch.stack((sqrt_OP, torch.zeros((), dtype=z.dtype, device=z.device)))
    )

    seg2_size = segcoord_2.shape[0]
    tmp = torch.stack((sqrt_OP, torch.zeros((), dtype=z.dtype, device=z.device)))
    coordinate_2 = []
    for i in range(1, seg2_size - 1):
        coordinate_2.append(tmp + torch.matmul(Rsigma2, segcoord_2[i]))

    seg3_size = segcoord_3.shape[0]
    tmp = torch.stack((xq, yq))
    coordinate_3 = [tmp]
    for i in range(1, seg3_size - 1):
        coordinate_3.append(tmp + torch.matmul(Rsigma3, segcoord_3[i]))

    coordinate = torch.stack(coordinate_1 + coordinate_2 + coordinate_3)
    coordinate = (coordinate - torch.mean(coordinate, dim=0)).to(z.device)

    # Original code `atan2(x, y)` is unusual, but preserved
    phig = torch.atan2(coordinate[0][0], coordinate[0][1])
    Rphig = _rotation_matrix(phig).to(z.device)
    coordinate = torch.stack(
        [
            torch.cat((torch.matmul(Rphig, coordinate[i]) * (-1), z[i].unsqueeze(0)))
            for i in range(ringsize)
        ]
    )
    return coordinate


def set_ring_pucker_coords(
    amplitude: torch.Tensor,
    angle: torch.Tensor,
    init_bondl: torch.Tensor,
    init_bondang: torch.Tensor,
    clip: bool = True,
) -> torch.Tensor:
    """Reconstruct 3D coordinates from puckering parameters.

    Args:
        amplitude: Tensor of CP amplitudes ``q``.
        angle: Tensor of CP phase angles ``phi``.
        init_bondl: Baseline bond lengths for the ring.
        init_bondang: Baseline bond angles for the ring.
        clip: Whether to clamp intermediate cosine values to ``[-1, 1]``.

    Returns:
        Tensor of shape ``(ringsize, 3)`` representing the reconstructed ring.

    Raises:
        ValueError: If any reconstruction step fails (e.g., invalid lengths,
            angles, or NaN coordinates).
    """
    N = init_bondl.shape[0]

    # 1. Get Z-coordinates
    newZ = get_z(N, amplitude, angle)

    # 2. Get 2D bond lengths
    #    This will raise ValueError if it fails
    new_bondl = _update_bond_length(N, newZ, init_bondl)

    # 3. Get 2D bond angles
    #    This will raise ValueError if it fails
    new_bondang = _update_beta(N, newZ, init_bondl, init_bondang, clip)

    # 4. Partition and reconstruct 3D coordinates
    newcoord = _ring_partition(N, newZ, new_bondl, new_bondang)

    if torch.isnan(newcoord).any():
        raise ValueError("Invalid coordinates after partitioning (NaN found)")

    return newcoord
