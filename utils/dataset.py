"""Dataset helpers used for PuckerFlow training and inference."""

import copy
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from utils.constants import BONDS_DICT_INV

from diffusion.sampling import PuckeringNoiseTransform
from utils.coordinate_transforms import (
    ab_to_cp,
    ab_to_x,
    fourier_position,
    import_dicts,
    set_pos,
    x_to_ab,
)

import utils.ring_canonicalization as ring_canon
from utils.featurization import featurize_mol, featurize_mol_from_smiles

LOGGER = logging.getLogger(__name__)

# Suppress RDKit logs
RDLogger.DisableLog("rdApp.*")  # type: ignore

BIG_OFFSET = 1000  # any large number that avoids collision with atom types

# --- Custom Exception Classes ---


class FeaturizationError(Exception):
    """Base class for errors during featurization."""

    pass


class RingTooLargeError(FeaturizationError):
    """Raised when a ring's size exceeds the processing limit."""

    pass


class MoleculeConstructionError(FeaturizationError):
    """Raised when RDKit fails to construct a molecule."""

    pass


class FeaturizationFailureError(FeaturizationError):
    """Raised when the featurization function fails."""

    pass


class CanonicalNameMismatchError(FeaturizationError):
    """Raised when canonical SMILES do not match."""

    pass


class InvalidPositionError(FeaturizationError):
    """Raised when generated positions are invalid (e.g., from ab_to_x)."""

    pass


# --- Datasets ---


class ConformerDataset(Dataset):
    """PyTorch Geometric dataset that provides cached ring conformers."""

    def __init__(
        self,
        config: Dict[str, Any],
        mode: str,
        transform: Optional[BaseTransform] = None,
    ):
        """Initialise the dataset for ``mode`` and ensure cache is ready."""
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol
        super().__init__("fake_root", transform)

        self.config = config
        self.cache_path = config["all_confs_path"]
        self.split_type = config["split_type"]
        self.bl_dict, self.bang_dict = import_dicts(config)
        self.seed = config["seed"]

        if not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            LOGGER.info("Preprocessing split %s", self.split_type)
            conf_dict = self.preprocess_datapoints(config)
            LOGGER.info("Caching conformers at %s", self.cache_path)

            all_datapoints: Dict[str, List[Data]] = {}
            for key in tqdm(conf_dict.keys(), desc="Collecting conformers"):
                valid_datapoints = [d for d in conf_dict[key] if "failed" not in d.mode]
                if valid_datapoints:
                    all_datapoints[key] = valid_datapoints

            # Save all datapoints in one pickle file
            LOGGER.info("Saving all datapoints in %s", self.cache_path)
            with open(self.cache_path, "wb") as f:
                pickle.dump(all_datapoints, f)
        else:
            LOGGER.info(
                "Reusing cached conformers at %s for mode %s",
                config["cache"],
                mode,
            )

        # Load all conformers from one pickle
        with open(self.cache_path, "rb") as f:
            all_data: Dict[str, List[Data]] = pickle.load(f)

        # Load the appropriate split JSON file
        with open(config["split_json_path"], "r") as f:
            split_smiles: List[str] = json.load(f)[mode]

        # Load datapoints for the current mode
        self.datapoints: List[Data] = []
        for i, smi in enumerate(split_smiles):
            if config["max_train_smiles"] > 0 and i >= config["max_train_smiles"]:
                break

            mol = Chem.MolFromSmiles(smi)
            if not mol:
                logging.warning(f"Could not parse SMILES: {smi}")
                continue

            n_atoms = mol.GetNumAtoms()
            if smi in all_data and 4 < n_atoms < 9:
                # Take up to 100 conformers per SMILES
                self.datapoints.extend(all_data[smi][:100])

    def preprocess_datapoints(self, config: Dict[str, Any]) -> Dict[str, List[Data]]:
        """Read the raw CSV, featurise every row, and group by SMILES."""
        # Merge dataset files
        try:
            data = pd.read_csv(config["dataset_csv"], header=0)
        except Exception as e:
            LOGGER.error("Failed to read dataset CSV %s: %s", config["dataset_csv"], e)
            exit(1)

        # Specify start indices for columns
        try:
            self.atom_start_idx = int(data.columns.get_loc("atom0"))
            self.bond_start_idx = int(data.columns.get_loc("bond0"))
            self.ab_start_idx = int(data.columns.get_loc("ab0"))
            self.size_idx = int(data.columns.get_loc("size"))
            self.filename_idx = int(data.columns.get_loc("filename"))
            self.smiles_idx = int(data.columns.get_loc("smiles"))
            self.mode_idx = int(data.columns.get_loc(f"{self.split_type}_{self.seed}"))
        except KeyError as e:
            LOGGER.error("Missing required column in CSV: %s", e)
            exit(1)

        LOGGER.info("Preparing to process %s datapoints", data.shape[0])

        # Do preprocessing
        errors: Dict[str, int] = defaultdict(int)
        conf_dict: Dict[str, List[Data]] = defaultdict(list)
        error_types = (
            RingTooLargeError,
            MoleculeConstructionError,
            FeaturizationFailureError,
            CanonicalNameMismatchError,
            InvalidPositionError,
            ValueError,
        )

        data_rows = data.values.tolist()

        with tqdm(total=len(data_rows), desc="Preprocessing dataset") as pbar:
            for row in data_rows:
                try:
                    t = self.featurize_mol(row)
                    conf_dict[t.name].append(t)
                except error_types as e:
                    errors[type(e).__name__] += 1
                pbar.update()

        if errors:
            LOGGER.info("Preprocessing error summary")
        else:
            LOGGER.info("No errors encountered during preprocessing")

        for err_type, count in errors.items():
            LOGGER.info("%s: %s", err_type, count)

        return conf_dict

    def len(self) -> int:
        """Returns the number of datapoints in the dataset."""
        return len(self.datapoints)

    def get(self, idx: int) -> Data:
        """Return a deep copy of datapoint ``idx`` to avoid mutation."""
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def featurize_mol(self, row: List[Any]) -> Data:
        """Convert a CSV row into a fully annotated :class:`Data` object."""
        N = int(row[self.size_idx])
        if N > 8:
            raise RingTooLargeError(f"Ring size {N} exceeds maximum allowed (8).")

        atoms = list(map(int, row[self.atom_start_idx : self.atom_start_idx + N]))
        bonds = row[self.bond_start_idx : self.bond_start_idx + N]

        ab_size = N - 3
        mol = Chem.RWMol()

        [mol.AddAtom(Chem.Atom(a)) for a in atoms]
        [
            mol.AddBond(i - 1, i % N, BONDS_DICT_INV[b])
            for i, b in enumerate(bonds, start=1)
        ]

        mol_obj = mol.GetMol()
        if mol_obj is None:
            raise MoleculeConstructionError(
                "Failed to construct molecule from atoms and bonds."
            )

        data = featurize_mol(mol_obj, self.seed)
        if not data:
            raise FeaturizationFailureError("Featurization of molecule failed.")

        data.name = Chem.CanonSmiles(Chem.MolToSmiles(data.mol))
        if data.name != row[self.smiles_idx]:
            LOGGER.warning(
                f"Canonical SMILES mismatch: expected {row[self.smiles_idx]}, got {data.name}"
            )
            raise CanonicalNameMismatchError(
                f"Canonical SMILES '{data.name}' does not match expected '{row[self.smiles_idx]}'."
            )

        # Generate bond length & angle keys
        length_keys_raw = [
            sorted([atoms[i], atoms[(i + 1) % N], i + BIG_OFFSET]) for i in range(N)
        ]
        length_keys = [
            f"{lk[0]:02}{bonds[lk[2]-BIG_OFFSET]}{lk[1]:02}{N:02}"
            for lk in length_keys_raw
        ]

        angle_keys_raw = []
        for i in range(N):
            ak = [
                atoms[i % N],
                bonds[i % N],
                atoms[(i + 1) % N],
                bonds[(i + 1) % N],
                atoms[(i + 2) % N],
            ]
            angle_keys_raw.append(min(ak, ak[::-1]))
        angle_keys = [
            "".join(f"{x:02}" for x in ak) + f"{N:02}" for ak in angle_keys_raw
        ]

        # Get CP coordinates from dataset and calculate accompanying ab
        data.ab_gt = torch.tensor(
            [
                row[ampi]
                for ampi in range(self.ab_start_idx, self.ab_start_idx + ab_size)
            ],
            requires_grad=False,
        )
        data.bl = torch.tensor(
            [self.bl_dict[length_keys[idx]] for idx in range(N)], requires_grad=False
        )
        data.bang = torch.tensor(
            [self.bang_dict[angle_keys[idx]] for idx in range(N)], requires_grad=False
        )
        data.cp_amp, data.cp_ang = ab_to_cp(data.ab_gt)
        data.pos = ab_to_x(data.ab_gt, data.bl, data.bang, clip=True)
        data.mode = row[self.mode_idx]

        set_pos(data.mol, data.pos)

        # Get all renumbered ab coordinates
        all_abs: List[torch.Tensor] = []
        mol_atoms = data.mol.GetNumAtoms()
        renumberings = ring_canon.rearrangement(
            data.mol, list(range(mol_atoms)), all=True
        )
        if not isinstance(renumberings, list) or not all(
            isinstance(r, list) for r in renumberings
        ):
            raise TypeError(
                "ring_canonicalization.rearrangement(all=True) did not return List[List[int]]"
            )

        for renum in renumberings:
            mol_renum = Chem.RenumberAtoms(data.mol, renum)
            pos_list = [
                mol_renum.GetConformer().GetAtomPosition(aid) for aid in range(N)
            ]
            pos = fourier_position(torch.tensor(pos_list, dtype=torch.float))
            all_abs.append(x_to_ab(pos))

        data.ab_gt_all = torch.cat(all_abs)
        data.num_renumberings = len(renumberings)

        return data


class InferenceDataset(Dataset):
    """
    Pytorch Geometric Dataset for inference.
    Featurizes a single SMILES string N times.
    """

    def __init__(self, smiles: str, n_confs: int, config: Dict[str, Any]):
        """
        Args:
            smiles: The SMILES string to featurize.
            n_confs: The number of conformers to generate (datapoints).
            config: A dictionary containing configuration parameters.
        """
        super().__init__()
        self.smiles = smiles
        self.n_confs = n_confs
        self.config = config
        self.datapoints = self.preprocess_datapoints()

    def len(self) -> int:
        """Returns the number of datapoints (n_confs)."""
        return len(self.datapoints)

    def get(self, idx: int) -> Data:
        """
        Gets a single datapoint.

        Args:
            idx: The index of the datapoint.

        Returns:
            A deep copy of the PyG Data object.
        """
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def preprocess_datapoints(self) -> List[Data]:
        """
        Featurizes the SMILES string and copies it n_confs times.

        Returns:
            A list of PyG Data objects.
        """
        data = featurize_mol_from_smiles(self.smiles, config=self.config)
        if data is None:
            logging.error(f"Failed to featurize SMILES for inference: {self.smiles}")
            return []
        return [copy.deepcopy(data) for _ in range(self.n_confs)]


# --- Loader Function ---


def construct_loader(
    config: Dict[str, Any],
    modes: Union[str, Tuple[str, ...], List[str]] = ("train", "val"),
) -> Union[DataLoader, List[DataLoader]]:
    """
    Constructs and returns DataLoaders for specified modes.

    Args:
        config: The configuration dictionary.
        modes: A string or list/tuple of strings ('train', 'val', 'test').

    Returns:
        A single DataLoader if one mode is requested, otherwise a list
        of DataLoaders.
    """
    if isinstance(modes, str):
        modes = [modes]

    loaders, datasets = [], []
    transform: BaseTransform = PuckeringNoiseTransform()
    for mode in modes:
        shuffle = True if mode == "train" else False
        dataset = ConformerDataset(config, mode, transform=transform)
        loader = DataLoader(
            dataset=dataset, batch_size=config["batch_size"], shuffle=shuffle
        )

        loaders.append(loader)
        datasets.append(dataset)

    return loaders[0] if len(modes) == 1 else loaders
