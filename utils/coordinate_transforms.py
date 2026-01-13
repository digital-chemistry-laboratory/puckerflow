"""Coordinate transforms shared across PuckerFlow.

This module exposes helpers to move between the different puckering
representations used throughout the project:

* Cartesian coordinates ``x`` in Angstrom.
* Height displacements ``z`` relative to the mean plane.
* Fourier coefficients ``ab``.
* Cremer-Pople amplitudes/angles ``(cp_amp, cp_ang)``.

It also contains lookup utilities for standardized bond dictionaries as
well as geometry helpers that are reused by both training and evaluation
pipelines.
"""

import json
import logging
import re
from pathlib import Path
from collections import UserDict
from typing import Any, Dict, List, Tuple, Union, cast
from numpy.typing import NDArray

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

import utils.ring_canonicalization as ring_canon
import utils.ring_geometry as ring_geom
import utils.ring_reconstruction as RR

ParsedKey = List[Union[int, float]]
LOGGER = logging.getLogger(__name__)

ab_to_cp = ring_geom.ab_to_cp
z_to_ab = ring_geom.z_to_ab
fourier_position = ring_geom.fourier_position

def _parse_key(key: str) -> ParsedKey:
    """Parse a standardized dictionary key into structured components.

    The dictionaries used for bond lengths/angles encode several integers
    and floats inside a single string (e.g. ``"061.50605"``).  This helper
    extracts those values as a list with the proper numeric types.

    Args:
        key: Raw dictionary key.

    Returns:
        Parsed numeric components that can be compared algebraically.
    """

    parts = re.findall(r"\d\.\d|\d{2}", key)
    return [float(p) if "." in p else int(p) for p in parts]


def _unparse_key(parsed: ParsedKey) -> str:
    """Serialize :func:`parse_key` output back to the compact string form."""

    parts: List[str] = []
    for value in parsed:
        parts.append(f"{value:.1f}" if isinstance(value, float) else f"{value:02d}")
    return "".join(parts)


def _weighted_distance(k1: ParsedKey, k2: ParsedKey, penalty_int: int = 3) -> float:
    """Return a distance measure used to locate the nearest dictionary key.

    The cost treats integer mismatches as cheaper than floating-point
    mismatches because the standardized dictionaries tolerate variations in
    atom indices more than they tolerate variations in bond orders.
    """

    dist = 0.0
    for a, b in zip(k1, k2):
        diff = abs(a - b) + (a > b) * 0.1
        if not isinstance(a, int):
            dist += diff * penalty_int
        else:
            dist += diff
    return dist


def _resolve_dict_path(filename: str) -> Path:
    """Return the first existing path for ``filename`` inside data/dictionaries."""

    candidates = [
        Path("data") / "dictionaries" / filename,
        Path(__file__).resolve().parent.parent / "data" / "dictionaries" / filename,
        Path(__file__).resolve().parents[2] / "data" / "dictionaries" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    search_list = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {filename}. Looked in:\n{search_list}")


class ClosestDict(UserDict):
    """Dictionary wrapper that supports fuzzy lookups for standardized keys."""

    def __getitem__(self, key: str) -> Any:
        """Return the value for ``key`` or the closest alternative."""

        if key in self.data:
            return self.data[key]
        try:
            target_key = _parse_key(key)
            existing_keys = [_parse_key(k) for k in self.data]
            closest_parsed_key = min(
                existing_keys,
                key=lambda candidate: _weighted_distance(candidate, target_key),
            )
            fallback_key = _unparse_key(closest_parsed_key)
            LOGGER.debug("ClosestDict fallback: %s -> %s", key, fallback_key)
            return self.data[fallback_key]
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise KeyError(
                f"Key '{key}' not found and no suitable fallback could be found."
            ) from exc


def import_dicts(config: Dict[str, Any]) -> Tuple[ClosestDict, ClosestDict]:
    """Load standardized bond length and angle dictionaries.

    The dictionaries depend on the dataset split as well as the sampling seed
    used when compiling the statistics, therefore both values are extracted
    from ``config`` to select the correct shard.

    Args:
        config: Experiment or dataset configuration containing ``split_type``
            and ``seed`` keys.

    Returns:
        Tuple ``(bl_dict, bang_dict)`` providing fuzzy lookups via
        :class:`ClosestDict`.
    """
    split_type = config["split_type"]
    seed = config["seed"]

    lengths_path = _resolve_dict_path("standardized_lengths.json")
    with lengths_path.open("r") as f:
        bl_dict = ClosestDict(json.load(f)[f"{split_type}_{seed}"])
    angles_path = _resolve_dict_path("standardized_angles.json")
    with angles_path.open("r") as f:
        bang_dict = ClosestDict(json.load(f)[f"{split_type}_{seed}"])
    return bl_dict, bang_dict


def ab_to_x(
    ab: torch.Tensor, bl: torch.Tensor, bang: torch.Tensor, clip: bool = False
) -> torch.Tensor:
    """Convert Fourier coefficients into Cartesian coordinates.

    Args:
        ab: One-dimensional tensor of length ``N-3`` containing the Fourier
            coefficients.
        bl: Bond length tensor of shape ``(N,)``.
        bang: Bond angle tensor of shape ``(N,)``.
        clip: Whether the reconstruction should clip invalid geometries.

    Returns:
        Tensor of shape ``(N, 3)`` containing reconstructed coordinates.
    """
    cp_amp, cp_ang = ab_to_cp(ab)
    return cp_to_x(cp_amp, cp_ang, bl, bang, clip)


def cp_to_x(
    cp_amp: torch.Tensor,
    cp_ang: torch.Tensor,
    bl: torch.Tensor,
    bang: torch.Tensor,
    clip: bool = False,
) -> torch.Tensor:
    """Convert Cremer–Pople coordinates to Cartesian positions."""
    coords = RR.set_ring_pucker_coords(cp_amp, cp_ang, bl, bang, clip)
    return coords.float().to(cp_amp.device)


def ab_to_z(ab: torch.Tensor) -> torch.Tensor:
    """Convert Fourier coordinates into height displacements ``z``."""
    ringsize = ab.shape[0] + 3
    q, ang = ab_to_cp(ab)
    return RR.get_z(ringsize, q, ang)


def x_to_cp(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Cartesian coordinates to the Cremer–Pople representation."""
    ccoord = ring_geom.translate(pos)
    cp_amp, cp_ang = ring_geom.get_ring_pucker_coords(ccoord)
    return cp_amp, cp_ang


def x_to_ab(pos: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian coordinates to Fourier coefficients."""
    cp_amp, cp_ang = x_to_cp(pos)
    return cp_to_ab(cp_amp, cp_ang)


def cp_to_ab(cp_amp: torch.Tensor, cp_ang: torch.Tensor) -> torch.Tensor:
    """Convert Cremer–Pople amplitudes/angles to Fourier coefficients."""
    ab_coords = torch.empty(cp_amp.shape[0] + cp_ang.shape[0], device=cp_amp.device)

    ab_coords[0 : cp_ang.shape[0] * 2 : 2] = cp_amp[: cp_ang.shape[0]] * torch.cos(
        cp_ang
    )
    ab_coords[1::2] = cp_amp[: cp_ang.shape[0]] * torch.sin(cp_ang)
    if cp_amp.shape[0] > cp_ang.shape[0]:
        ab_coords[-1] = cp_amp[-1]

    return ab_coords


def mol_to_abs(mol: Chem.Mol) -> List[torch.Tensor]:
    """Calculate Fourier coordinates for every canonical renumbering."""
    all_abs: List[torch.Tensor] = []
    N = mol.GetNumAtoms()
    renumberings = ring_canon.rearrangement(mol, list(range(N)), all=True)
    renumberings = cast(List[List[int]], renumberings)

    for renumbering in renumberings:
        mol_renum = Chem.RenumberAtoms(mol, renumbering)
        pos_list = [mol_renum.GetConformer().GetAtomPosition(aid) for aid in range(N)]
        pos = fourier_position(torch.tensor(pos_list, dtype=torch.float))
        all_abs.append(x_to_ab(pos))
    return all_abs


def set_pos(mol: Chem.Mol, pos: torch.Tensor) -> None:
    """Set all conformer coordinates on ``mol`` from ``pos``."""
    np_pos = pos.cpu().detach().numpy().astype(np.double)
    for j in range(mol.GetNumAtoms()):
        mol.GetConformer().SetAtomPosition(j, np_pos[j].tolist())


def get_pos(mol: Chem.Mol) -> torch.Tensor:
    """Return the conformer positions for ``mol`` as a tensor."""
    return torch.tensor(mol.GetConformer().GetPositions())


def perturb_cp_instance(
    data: Data, ab_updates: torch.Tensor, reset: bool = False, clip: bool = True
) -> Tuple[Union[torch.Tensor, int], ...]:
    """Apply ``ab`` perturbations and reconstruct coordinates.

    Args:
        data: PyG data instance that contains ``ab``, ``bl`` and ``bang``
            attributes.
        ab_updates: Vector of updates (or replacement values when ``reset`` is
            ``True``).
        reset: Whether to overwrite ``data.ab`` instead of adding the updates.
        clip: Propagate ``clip`` flag to :func:`cp_to_x`.

    Returns:
        Tuple ``(cp_amp_new, cp_ang_new, ab, pos)``. When reconstruction fails,
        the tuple contains sentinel integers originating from the reconstruction
        helpers.
    """
    if reset:
        ab = ab_updates.to(data.x.device)
    else:
        ab = data.ab + ab_updates.to(data.x.device)

    cp_amp_new, cp_ang_new = ab_to_cp(ab)
    pos = cp_to_x(cp_amp_new, cp_ang_new, data.bl, data.bang, clip=clip)

    if clip:
        # Re-calculate cp/ab from the potentially clipped 3D positions
        cp_amp_new, cp_ang_new = x_to_cp(pos)
        ab = cp_to_ab(cp_amp_new, cp_ang_new)

    return cp_amp_new, cp_ang_new, ab, pos


def z_to_ab_batch(data: Batch, out: torch.Tensor) -> torch.Tensor:
    """Vectorised ``z`` to ``ab`` conversion for PyG batches."""
    z_list = [
        out.reshape(-1)[data.ptr[i] : data.ptr[i + 1]]
        for i in range(data.ptr.shape[0] - 1)
    ]
    ab_list = [ring_geom.z_to_ab(z) for z in z_list]
    return torch.cat(ab_list)


# --- Geometry Functions ---


def get_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Returns the angle (a-b-c) between 0 and pi.

    Args:
        a, b, c: 3D points as numpy arrays.

    Returns:
        The angle in radians.
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.arccos(cosine_angle))


def ptv(point: Chem.rdGeometry.Point3D) -> NDArray[np.float64]:
    """
    Converts an RDKit Point3D to a numpy array.

    Args:
        point: An RDKit Point3D object.

    Returns:
        A 3D numpy array.
    """
    arr = np.array([float(point.x), float(point.y), float(point.z)], dtype=np.float64)
    return cast(NDArray[np.float64], arr)


def invert_coords(mol: Chem.Mol) -> None:
    """
    Inverts the coordinates of a molecule's conformer (pos -> -pos).

    Args:
        mol: RDKit molecule to modify.
    """
    set_pos(mol, -get_pos(mol))


def mol_to_z(mol: Chem.Mol) -> torch.Tensor:
    """
    Calculates the 'z' coordinates for a molecule's canonical renumbering.

    Args:
        mol: An RDKit molecule with a 3D conformer.

    Returns:
        A tensor of 'z' coordinates (the 3rd dim of Fourier positions).
    """
    N = mol.GetNumAtoms()
    Chem.RemoveStereochemistry(mol)
    renumbering = ring_canon.rearrangement(mol, list(range(N)), all=False)
    renumbering = cast(List[int], renumbering)  # Ensure mypy knows type

    mol_renum = Chem.RenumberAtoms(mol, renumbering)
    pos_list = [mol_renum.GetConformer().GetAtomPosition(aid) for aid in range(N)]
    fourier_pos = fourier_position(torch.tensor(pos_list, dtype=torch.float))

    # Return just the z-coordinates (3rd column)
    return fourier_pos[:, 2]
