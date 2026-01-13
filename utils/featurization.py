"""
Featurization pipeline to convert RDKit molecules into
torch_geometric.data.Data objects for graph neural networks.

Includes featurization of atoms, bonds, and ring-specific properties
like ideal bond lengths (bl) and bond angles (bang).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

import utils.ring_canonicalization as ring_canon
from utils.coordinate_transforms import import_dicts
from utils.constants import BONDS_DICT, BOND_INDICES

LOGGER = logging.getLogger(__name__)

# --- Helper Functions ---


def one_k_encoding(value: Any, choices: List[Any]) -> List[int]:
    """Return one-hot encoding with an additional "other" bin."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def one_k_encoding_inc(value: Any, choices: List[Any]) -> List[int]:
    """Return one-hot encoding without an "other" bin."""
    encoding = [0] * len(choices)
    index = choices.index(value)
    encoding[index] = 1
    return encoding


# --- Core Featurization ---


def featurize_mol(mol: Chem.Mol, seed: int) -> Data:
    """Create a :class:`torch_geometric.data.Data` object from ``mol``."""
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        # Generate a 3D conformer, required for some downstream steps
        AllChem.EmbedMolecule(mol, randomSeed=seed)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Sanitize/RemoveHs/EmbedMolecule failed for "
            f"{Chem.MolToSmiles(mol)}: {e}"
        ) from e

    atom_features: List[List[Union[int, float]]] = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(Chem.Mol.GetAtoms(mol)):
        features: List[Union[int, float]] = []
        features.extend(
            one_k_encoding(atom.GetAtomicNum(), [5, 6, 7, 8, 14, 15, 16, 34])
        )
        features.extend(
            one_k_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )
        )
        features.extend(
            one_k_encoding_inc(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        )
        features.extend(
            [
                int(ring.IsAtomInRingOfSize(i, 5)),
                int(ring.IsAtomInRingOfSize(i, 6)),
                int(ring.IsAtomInRingOfSize(i, 7)),
                int(ring.IsAtomInRingOfSize(i, 8)),
            ]
        )

        features.extend(one_k_encoding_inc(i, list(range(8))))
        atom_features.append(features)

    row, col, edge_type = [], [], []
    for bond in Chem.Mol.GetBonds(mol):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS_DICT[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    # Convert float edge types (like 1.5) to integer indices for one-hot
    edge_type_indices = torch.tensor(
        [BOND_INDICES[t] for t in edge_type], dtype=torch.long
    )
    edge_attr = F.one_hot(edge_type_indices, num_classes=len(BOND_INDICES)).to(
        torch.float
    )

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol=mol,
    )


def featurize_mol_from_smiles(smiles: str, config: Dict[str, Any]) -> Optional[Data]:
    """Full featurization pipeline for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        LOGGER.warning("Could not create molecule from SMILES: %s", smiles)
        return None

    try:
        bl_dict, bang_dict = import_dicts(config)
    except ImportError:
        LOGGER.error("Could not import coordinate_transforms or find dictionaries.")
        return None
    except Exception as e:
        LOGGER.error("Failed to load bl/bang dicts: %s", e)
        return None

    N = mol.GetNumAtoms()

    try:
        # Get canonical atom order
        canonical_order = ring_canon.rearrangement(mol, list(range(N)), all=False)
        mol = rdmolops.RenumberAtoms(mol, canonical_order)
        # Embed molecule *after* renumbering
        AllChem.EmbedMolecule(mol, randomSeed=config["seed"])  # type: ignore

        data = featurize_mol(mol, config["seed"])
    except Exception as e:
        LOGGER.warning("Failed to featurize %s: %s", smiles, e)
        return None

    data.name = smiles
    data.mol = mol  # This is the renumbered mol
    data.N = N

    atoms = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)]
    data.atoms = atoms

    bonds = [
        mol.GetBondBetweenAtoms(i, (i + 1) % N).GetBondTypeAsDouble() for i in range(N)
    ]
    data.bonds = bonds

    # These keys are complex strings used to look up ideal bond/angle values
    data.length_keys = [
        f"{sorted([atoms[i], atoms[(i+1)%N], i+100])[0]:02}"
        f"{bonds[i]:.1f}{sorted([atoms[i], atoms[(i+1)%N], i+100])[1]:02}{N:02}"
        for i in range(N)
    ]

    angle_keys_forward = [
        [
            atoms[i % N],
            bonds[i % N],
            atoms[(i + 1) % N],
            bonds[(i + 1) % N],
            atoms[(i + 2) % N],
        ]
        for i in range(N)
    ]
    angle_keys = [
        "".join(f"{x:02}" for x in min(ak, ak[::-1])) + f"{N:02}"
        for ak in angle_keys_forward
    ]

    try:
        data.bl = torch.tensor(
            [bl_dict[key] for key in data.length_keys], requires_grad=False
        )
        data.bang = torch.tensor(
            [bang_dict[key] for key in angle_keys], requires_grad=False
        )
    except KeyError as e:
        LOGGER.warning("Missing appropriate bond/angle key for %s: %s", smiles, e)
        return None

    return data
