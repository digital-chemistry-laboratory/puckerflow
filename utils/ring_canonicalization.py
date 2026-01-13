"""Canonical ring ordering utilities for PuckerFlow.

These helpers determine a deterministic atom ordering for monocyclic rings by
scoring bond patterns and atomic numbers, ensuring consistent featurisation and
augmentation.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def get_ring_element(mol: Chem.Mol, idxlist: List[int]) -> List[Tuple[int, int]]:
    """Return atomic numbers for the provided indices.

    Args:
        mol: RDKit molecule containing the ring.
        idxlist: Atom indices that make up the ring path.

    Returns:
        List of ``(atom_idx, atomic_number)`` pairs.
    """

    return [(node, mol.GetAtomWithIdx(node).GetAtomicNum()) for node in idxlist]


def get_ring_bonds(
    mol: Chem.Mol, ring_path: List[int]
) -> List[Tuple[Tuple[int, int], float]]:
    """Return bond types for every adjacent pair along the ring path.

    Args:
        mol: RDKit molecule containing the ring.
        ring_path: Atom indices describing a continuous ring.

    Returns:
        List of ``((idx_a, idx_b), bond_order)`` tuples where ``idx_a < idx_b``.
    """

    ringsize = len(ring_path)
    bond_orders = [
        float(
            mol.GetBondBetweenAtoms(
                ring_path[i], ring_path[(i + 1) % ringsize]
            ).GetBondType()
        )
        for i in range(ringsize)
    ]
    bond_orders = [1.5 if b == 12 else b for b in bond_orders]
    bond_pairs = [
        (
            (ring_path[i], ring_path[(i + 1) % ringsize])
            if ring_path[i] <= ring_path[(i + 1) % ringsize]
            else (ring_path[(i + 1) % ringsize], ring_path[i])
        )
        for i in range(ringsize)
    ]
    return list(zip(bond_pairs, bond_orders))


def enumerate_atom_orders(
    idxlist: List[int], ringsize: int, init_index: List[int]
) -> List[List[int]]:
    """Enumerate clockwise and anticlockwise atom orderings.

    Args:
        idxlist: Canonical ring path indices.
        ringsize: Number of atoms in the ring.
        init_index: Seed indices corresponding to maximal bond scores.

    Returns:
        List of atom orders (clockwise and anticlockwise) starting at the
        provided seeds.
    """

    all_orders = []
    idxarr = torch.tensor(idxlist)
    for i in init_index:
        clock = np.mod(list(range(i, i + ringsize)), ringsize)
        anti = np.mod(list(range(i + 1, i + 1 - ringsize, -1)), ringsize)
        all_orders.append(idxarr[clock].tolist())
        all_orders.append(idxarr[anti].tolist())
    return all_orders


def get_neighbours(mol: Chem.Mol, idx: int) -> Tuple[List[int], List[int]]:
    """Return neighbors and bond types for a given atom.

    Args:
        mol: RDKit molecule containing the atom.
        idx: Atom index for which to fetch neighbors.

    Returns:
        Tuple of neighbor indices and their associated integer bond orders.
    """

    connected_atoms = mol.GetAtomWithIdx(idx).GetNeighbors()
    atomidx = [atom.GetIdx() for atom in connected_atoms]
    bonds = [int(mol.GetBondBetweenAtoms(idx, x).GetBondType()) for x in atomidx]
    return atomidx, bonds


def sorting(score_frame: pd.DataFrame, size: int) -> pd.Index:
    """Filter candidate orderings by bond/connectivity/element scores.

    Args:
        score_frame: DataFrame containing cumulative bond and element scores.
        size: Number of atoms in the ring (half the number of columns).

    Returns:
        Index into ``score_frame`` for the retained orderings.
    """

    midpoint = size
    bonds = score_frame.iloc[:, :midpoint]
    bmax = max(bonds.iloc[:, midpoint - 1])
    selected = score_frame[score_frame.iloc[:, midpoint - 1] == bmax]
    for i in range(0, midpoint - 1):
        colmax = max(selected.iloc[:, i])
        selected = selected[selected.iloc[:, i] == colmax]

    element_frame = score_frame.iloc[selected.index, size : size + midpoint]
    emin = min(element_frame.iloc[:, midpoint - 1])
    selected = element_frame[element_frame.iloc[:, midpoint - 1] == emin]
    for i in range(0, midpoint - 1):
        colmin = min(selected.iloc[:, i])
        selected = selected[selected.iloc[:, i] == colmin]
    return selected.index


def rearrangement(
    mol: Chem.Mol, idxlist: List[int], all: bool = False
) -> Union[List[int], List[List[int]]]:
    """Derive canonical ring atom ordering(s).

    Args:
        mol: RDKit molecule containing the ring of interest.
        idxlist: Atom indices defining the ring path.
        all: Whether to return every equally scoring ordering.

    Returns:
        Either a single ordering when ``all`` is ``False`` or a list of
        orderings otherwise.

    Raises:
        ValueError: If the ring path cannot be traversed continuously.
        RuntimeError: If ranking fails to produce any ordering.
    """

    ringsize = len(idxlist)
    ringloop = [idxlist[0]]
    for _ in range(ringsize - 1):
        atomidx, _ = get_neighbours(mol, ringloop[-1])
        checklist = [x for x in atomidx if x in idxlist and x not in ringloop]
        if not checklist:
            raise ValueError(f"Ring path is broken or incomplete for ring {Chem.MolToSmiles(mol)} (ringsize={ringsize}).")
        ringloop.append(checklist[0])

    ring_bonds = get_ring_bonds(mol, ringloop)
    bondarray = torch.tensor([b[1] for b in ring_bonds])
    bonddict = dict(ring_bonds)
    eledict = dict(get_ring_element(mol, ringloop))
    maximum = max(bondarray)
    init_indx = [i for i, j in enumerate(bondarray) if j == maximum]
    orders = enumerate_atom_orders(ringloop, ringsize, init_indx)

    bond_scores, element_scores = [], []
    for order in orders:
        ordered_bonds: List[Tuple[int, int]] = []
        for x in range(ringsize):
            idx1 = order[x % ringsize]
            idx2 = order[(x + 1) % ringsize]
            ordered_bonds.append((idx1, idx2) if idx1 <= idx2 else (idx2, idx1))
        bond_scores.append([bonddict[b] for b in ordered_bonds])
        element_scores.append([eledict[atom] for atom in order])

    eframe = pd.DataFrame(
        element_scores, columns=[f"E{x}" for x in range(ringsize)]
    ).cumsum(axis=1)
    bondframe = pd.DataFrame(
        bond_scores, columns=[f"B{x}" for x in range(ringsize)]
    ).cumsum(axis=1)
    dataframe = pd.concat([bondframe, eframe], axis=1)
    index = sorting(dataframe, ringsize)

    if len(index) >= 1 and not all:
        return orders[index[0]]
    if all:
        return [orders[i] for i in index]
    raise RuntimeError("Sorting returned no valid index.")


def get_num_rearrangements(smiles: str) -> int:
    """Return the number of equivalent canonical rearrangements for SMILES.

    Args:
        smiles: Molecule represented as a SMILES string.

    Returns:
        Number of canonical rearrangements recovered for the ring.

    Raises:
        ValueError: If the SMILES string cannot be parsed.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return len(rearrangement(mol, list(range(mol.GetNumAtoms())), all=True))
