"""Centralized constants shared across PuckerFlow pipeline."""

from typing import Dict
from rdkit.Chem.rdchem import BondType as BT

BONDS_DICT: Dict[BT, float] = {
    BT.SINGLE: 1.0,
    BT.DOUBLE: 2.0,
    BT.TRIPLE: 3.0,
    BT.AROMATIC: 1.5,
}
"""Map :class:`rdchem.BondType` to the numeric bond order."""

BONDS_DICT_INV: Dict[float, BT] = {value: key for key, value in BONDS_DICT.items()}
"""Map numeric bond order to :class:`rdchem.BondType`."""

BOND_INDICES: Dict[float, int] = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}
"""Map bond order to one-hot index."""

N_max = 8
"""Maximum number of atoms supported in a puckering ring."""

N_ab_max = N_max - 3
"""Number of Cremer-Pople coefficients inferred from ``N_max``."""

N_max_confs = 1000
"""Upper bound on conformers processed for a molecule."""

SIZE_SPLIT = 6
"""Ring size for unseen size split generalization experiment."""
