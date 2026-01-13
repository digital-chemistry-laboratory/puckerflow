"""Utility functions for model setup, training, and data handling."""

# 1. Standard Library Imports
import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, TypeVar, Union

# 2. Third-Party Imports
import numpy as np
import torch
import yaml
from rdkit import Chem
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm

# 3. Local Application Imports
from diffusion.score_model import TensorProductScoreModel

ConfigDict = Dict[str, Any]
LRScheduler = Union[_LRScheduler, ReduceLROnPlateau]
T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG used by the training stack.

    Args:
        seed: Integer used to seed Python's ``random`` module, NumPy,
            ``torch`` (CPU and CUDA), and cuDNN determinism flags.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(config: ConfigDict) -> TensorProductScoreModel:
    """Instantiate :class:`TensorProductScoreModel` from configuration data.

    Args:
        config: Mapping created from ``yaml`` configuration files. Must contain
            the architectural hyperparameters consumed by
            :class:`TensorProductScoreModel`.

    Returns:
        A fully constructed score model ready for training or evaluation.
    """
    return TensorProductScoreModel(
        in_node_features=config["in_node_features"],
        sigma_embed_dim=config["sigma_embed_dim"],
        ns=config["ns"],
        nv=config["nv"],
        num_conv_layers=config["num_conv_layers"],
        max_radius=config["max_radius"],
        radius_embed_dim=config["radius_embed_dim"],
        use_second_order_repr=config["use_second_order_repr"],
        batch_norm=not config["no_batch_norm"],
        residual=not config["no_residual"],
    )


def get_optimizer_and_scheduler(
    config: ConfigDict, model: torch.nn.Module
) -> Tuple[Optimizer, LRScheduler]:
    """Create the AdamW optimizer and plateau scheduler used in training.

    Args:
        config: Training configuration with ``lr`` and ``scheduler_patience``
            keys.
        model: Neural network whose parameters need to be optimized.

    Returns:
        Tuple containing the optimizer and LR scheduler.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=config["scheduler_patience"],
        min_lr=config["lr"] / 100,
    )
    return optimizer, scheduler


def save_yaml_file(path: Union[str, Path], content: ConfigDict) -> None:
    """Serialize a configuration dictionary to disk.

    Args:
        path: Output path for the YAML file.
        content: Arbitrary mapping that should be dumped with ``yaml``.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving YAML file to %s", target)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data=content, stream=handle, sort_keys=False)


def merge_dicts(
    *dicts: Dict[str, T], nested: bool = False
) -> Dict[str, Union[List[T], Dict[str, List[T]]]]:
    """Merge dictionaries by aggregating values into lists.

    Args:
        *dicts: Mapping objects to merge.
        nested: When ``True`` expect each value to be a dictionary whose
            contents should be merged independently. The returned structure is
            two levels deep in that case.

    Returns:
        Dictionary whose values are lists collecting entries from ``dicts``.
    """

    if nested:
        merged_nested: DefaultDict[str, DefaultDict[str, List[T]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for mapping in dicts:
            for key, nested_dict in mapping.items():
                if not isinstance(nested_dict, dict):
                    raise TypeError("nested=True expects dict values")
                for inner_key, value in nested_dict.items():
                    merged_nested[key][inner_key].append(value)
        return {key: dict(inner) for key, inner in merged_nested.items()}

    merged: DefaultDict[str, List[T]] = defaultdict(list)
    for mapping in dicts:
        for key, value in mapping.items():
            merged[key].append(value)
    return dict(merged)


def nested_defaultdict() -> DefaultDict[Any, List[Any]]:
    """Return ``defaultdict(list)`` with improved type inference for callers."""
    return defaultdict(list)


def get_error(exception: Exception) -> str:
    """Map RDKit exception strings to short human-readable categories.

    Args:
        exception: Exception raised by RDKit or downstream geometry routines.

    Returns:
        Readable string describing the likely failure mode.
    """

    error = str(exception)
    mapping = {
        "SanitizeMol": "SanitizeMol failed",
        "EmbedMolecule returned": "EmbedMolecule failed",
        "EmbedMolecule changed": "EmbedMolecule changed hybridization or bond types",
        "Invalid bond lengths": "Bond lengths error",
        "Invalid bond angles": "Bond angles error",
        "Invalid coordinates after partitioning": "Partitioning error",
    }
    for key, message in mapping.items():
        if key in error:
            return message
    return error

def convert_pickles_to_sdf(config: ConfigDict) -> None:
    """Export cached conformers into per-SMILES SDF files and name lists.

    Args:
        config: Configuration containing ``sdf_dir``, ``all_confs_path``,
            ``split_json_path``, ``split_type``, and ``seed`` entries.
    """

    required_keys = (
        "sdf_dir",
        "all_confs_path",
        "split_json_path",
        "split_type",
        "seed",
    )
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    sdf_output_dir = Path(str(config["sdf_dir"]))
    if sdf_output_dir.exists():
        for file_path in sdf_output_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    else:
        sdf_output_dir.mkdir(parents=True, exist_ok=True)

    with Path(str(config["all_confs_path"])).open("rb") as handle:
        all_confs_dict: Dict[str, List[Data]] = pickle.load(handle)
    with Path(str(config["split_json_path"])).open("r", encoding="utf-8") as handle:
        split_smiles_dict: Dict[str, List[str]] = json.load(handle)

    should_write_sdf = config.get("seed") == 42

    for mode in ("train", "test", "val"):
        smiles_list = split_smiles_dict.get(mode, [])
        complex_names: List[str] = []
        conformer_index = 0
        writer: Optional[Chem.SDWriter] = None

        for smiles_name in tqdm(smiles_list, desc=f"Processing {mode}"):
            data_list = all_confs_dict.get(smiles_name)
            if data_list is None:
                LOGGER.warning("%s missing from cached conformers", smiles_name)
                continue

            if should_write_sdf:
                sdf_file_path = sdf_output_dir / f"{smiles_name}.sdf"
                writer = Chem.SDWriter(str(sdf_file_path))

            for data in data_list:
                mol = getattr(data, "mol", None)
                if mol is None:
                    continue
                mol_name = f"{smiles_name}_{conformer_index}"
                conformer_index += 1
                if should_write_sdf and writer is not None:
                    writer.write(mol)
                complex_names.append(mol_name)

            if writer is not None:
                writer.close()
                writer = None

        txt_file_path = sdf_output_dir / (
            f"complexes-{mode}-{config['split_type']}_{config['seed']}.txt"
        )
        with txt_file_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(complex_names))
        LOGGER.info(
            "%s, %s_%s names written to: %s",
            mode.capitalize(),
            config["split_type"],
            config["seed"],
            txt_file_path,
        )

    LOGGER.info(
        "Conversion complete! All SDF files and name lists are written to %s",
        sdf_output_dir,
    )
