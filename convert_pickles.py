#!/usr/bin/env python3
"""
Unified pickle conversion script for safe publication.

This script converts pickle files containing RDKit molecules, numpy arrays,
torch tensors, and PyTorch Geometric data to safe, publishable formats and back.

Safe formats used:
- RDKit molecules: SDF files
- Numpy arrays/torch tensors: .npz files (compressed numpy archives)
- PyTorch Geometric Data: JSON metadata + .npz for tensor data
- Config dictionaries: JSON files

Usage:
    # Convert all pickles to safe format (for publishing)
    python convert_pickles.py to_safe

    # Convert safe format back to pickles (for users)
    python convert_pickles.py to_pickle

    # Verify a specific conversion
    python convert_pickles.py verify <original.pkl> <restored.pkl>

    # Convert a single file
    python convert_pickles.py to_safe --file <input.pkl> --output <output_dir>
    python convert_pickles.py to_pickle --input <input_dir> --output <output.pkl>
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.data import Data as PyGData

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from rdkit import Chem


# ============================================================================
# Configuration: Define all pickle files and their locations
# ============================================================================

SEEDS = [42, 43, 44, 45, 46]

# Define all pickle file patterns and their safe output locations
PICKLE_CONFIGS = {
    # MCF samples
    "mcf_samples": {
        "pattern": "comparison_algorithms/mcf_samples/ml-mcf_samples_random_split_{seed}.pkl",
        "safe_dir": "comparison_algorithms/mcf_samples/ml-mcf_samples_random_split_{seed}",
        "type": "samples_with_config",
    },
    # GeoDiff samples
    "geodiff_samples": {
        "pattern": "comparison_algorithms/geodiff_samples/geodiff_samples_random_split_{seed}.pkl",
        "safe_dir": "comparison_algorithms/geodiff_samples/geodiff_samples_random_split_{seed}",
        "type": "samples_with_config",
    },
    # RDKit ETKDG samples
    "rdkit_etkdg": {
        "pattern": "runs/rdkit_etkdg_{seed}.pkl",
        "safe_dir": "runs/safe_rdkit_etkdg_{seed}",
        "type": "rdkit_samples",
    },
    # RDKit ETKDG small torsions
    "rdkit_etkdg_small_torsions": {
        "pattern": "runs/rdkit_etkdg_small_torsions_{seed}.pkl",
        "safe_dir": "runs/safe_rdkit_etkdg_small_torsions_{seed}",
        "type": "rdkit_samples",
    },
    # RDKit KDG samples
    "rdkit_kdg": {
        "pattern": "runs/rdkit_kdg_{seed}.pkl",
        "safe_dir": "runs/safe_rdkit_kdg_{seed}",
        "type": "rdkit_samples",
    },
    # PuckerFlow sample_confs
    "puckerflow_sample_confs": {
        "pattern": "runs/run_{seed}/conformers/sample_confs.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_sample_confs",
        "type": "puckerflow_samples",
    },
    # PuckerFlow sample_confs_with_relaxed
    "puckerflow_sample_confs_relaxed": {
        "pattern": "runs/run_{seed}/conformers/sample_confs_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_sample_confs_with_relaxed",
        "type": "puckerflow_samples",
    },
    # PuckerFlow ab_paths
    "puckerflow_ab_paths": {
        "pattern": "runs/run_{seed}/conformers/ab_paths.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths",
        "type": "numpy_dict",
    },
    # PuckerFlow benchmarking files
    "puckerflow_benchmarking_1": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_1.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_1",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_2": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_2.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_2",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_5": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_5.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_5",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_10": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_10.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_10",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_20": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_20.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_20",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_30": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_30.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_30",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_50": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_50.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_50",
        "type": "puckerflow_samples",
    },
    # Benchmarking with relaxed
    "puckerflow_benchmarking_1_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_1_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_1_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_2_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_2_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_2_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_5_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_5_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_5_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_10_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_10_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_10_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_20_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_20_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_20_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_30_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_30_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_30_with_relaxed",
        "type": "puckerflow_samples",
    },
    "puckerflow_benchmarking_50_relaxed": {
        "pattern": "runs/run_{seed}/conformers/benchmarking_50_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_benchmarking_50_with_relaxed",
        "type": "puckerflow_samples",
    },
    # AB paths benchmarking
    "puckerflow_ab_paths_benchmarking_1": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_1.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_1",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_2": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_2.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_2",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_5": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_5.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_5",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_10": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_10.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_10",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_20": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_20.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_20",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_30": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_30.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_30",
        "type": "numpy_dict",
    },
    "puckerflow_ab_paths_benchmarking_50": {
        "pattern": "runs/run_{seed}/conformers/ab_paths_benchmarking_50.pkl",
        "safe_dir": "runs/run_{seed}/conformers/safe_ab_paths_benchmarking_50",
        "type": "numpy_dict",
    },
    # Conformer dataset cache
    "conformer_dataset": {
        "pattern": "data/cache/conformer_dataset/all_conformers_random_split_{seed}.pkl",
        "safe_dir": "data/cache/conformer_dataset/safe_all_conformers_random_split_{seed}",
        "type": "pyg_dataset",
    },
    # PuckerFlow small model sample_confs
    "puckerflow_small_sample_confs": {
        "pattern": "runs/run_{seed}_small/conformers/sample_confs.pkl",
        "safe_dir": "runs/run_{seed}_small/conformers/safe_sample_confs",
        "type": "puckerflow_samples",
    },
    # PuckerFlow small model sample_confs_with_relaxed
    "puckerflow_small_sample_confs_relaxed": {
        "pattern": "runs/run_{seed}_small/conformers/sample_confs_with_relaxed.pkl",
        "safe_dir": "runs/run_{seed}_small/conformers/safe_sample_confs_with_relaxed",
        "type": "puckerflow_samples",
    },
    # PuckerFlow small model ab_paths
    "puckerflow_small_ab_paths": {
        "pattern": "runs/run_{seed}_small/conformers/ab_paths.pkl",
        "safe_dir": "runs/run_{seed}_small/conformers/safe_ab_paths",
        "type": "numpy_dict",
    },
}

# Non-seed files
NON_SEED_PICKLE_CONFIGS = {
    "orig_mol_dict": {
        "pattern": "data/orig_mol_dict.pkl",
        "safe_dir": "data/safe_orig_mol_dict",
        "type": "mol_dict",
    },
}


# ============================================================================
# Helper functions for type conversion
# ============================================================================


def sanitize_key(key: str) -> str:
    """Convert a key to a safe filename."""
    return key.replace("/", "_slash_").replace("\\", "_backslash_").replace("=", "_eq_")


def unsanitize_key(key: str) -> str:
    """Convert a safe filename back to original key."""
    return key.replace("_slash_", "/").replace("_backslash_", "\\").replace("_eq_", "=")


def numpy_to_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays and tensors to serializable format."""
    if isinstance(obj, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    elif HAS_TORCH and isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return {
            "__torch__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    elif isinstance(obj, dict):
        return {key: numpy_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def serializable_to_numpy(
    obj: Any, arrays: Dict[str, np.ndarray], prefix: str = ""
) -> Any:
    """Recursively convert serializable format back to numpy/torch arrays."""
    if isinstance(obj, dict):
        if obj.get("__numpy__") is True:
            return arrays[prefix]
        elif obj.get("__torch__") is True:
            arr = arrays[prefix]
            dtype_str = obj.get("dtype", "float32")
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
            }
            torch_dtype = dtype_map.get(dtype_str, torch.float32)
            return torch.tensor(arr, dtype=torch_dtype)
        else:
            return {
                key: serializable_to_numpy(
                    value, arrays, f"{prefix}.{key}" if prefix else key
                )
                for key, value in obj.items()
            }
    elif isinstance(obj, list):
        return [
            serializable_to_numpy(item, arrays, f"{prefix}[{i}]")
            for i, item in enumerate(obj)
        ]
    else:
        return obj


def collect_arrays(obj: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    """Collect all numpy arrays from a nested structure."""
    arrays = {}
    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
    elif HAS_TORCH and isinstance(obj, torch.Tensor):
        arrays[prefix] = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            arrays.update(collect_arrays(value, new_prefix))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            arrays.update(collect_arrays(item, f"{prefix}[{i}]"))
    return arrays


# ============================================================================
# Conversion functions for different data types
# ============================================================================


def convert_mol_list_to_sdf(
    mol_list: List[Chem.Mol], output_path: Path, key: str
) -> None:
    """Write a list of RDKit molecules to an SDF file with key stored as property."""
    writer = Chem.SDWriter(str(output_path))
    for i, mol in enumerate(mol_list):
        mol.SetProp("DICT_KEY", key)
        mol.SetProp("INDEX", str(i))
        writer.write(mol)
    writer.close()


def load_mol_list_from_sdf(sdf_path: Path) -> List[Chem.Mol]:
    """Load a list of RDKit molecules from an SDF file."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = []
    for mol in suppl:
        if mol is not None:
            mols.append(mol)
    return mols


# ============================================================================
# Type-specific converters: to_safe
# ============================================================================


def convert_samples_with_config_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert MCF/GeoDiff samples pickle to safe format.
    Structure: {smiles: {mol_relaxed: [Mol], ab_relaxed: [[]], mol_unrelaxed: [Mol], ab_unrelaxed: [[]], config: {}}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    molecules_dir = output_dir / "molecules"
    molecules_dir.mkdir(exist_ok=True)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(exist_ok=True)

    for smiles_key, entry in data.items():
        safe_key = sanitize_key(smiles_key)
        entry_meta = {}
        arrays_to_save = {}

        for sub_key, sub_value in entry.items():
            # Skip config key - don't include it in conversion
            if sub_key == "config":
                continue
            
            if (
                isinstance(sub_value, list)
                and len(sub_value) > 0
                and isinstance(sub_value[0], Chem.Mol)
            ):
                # Molecule list -> SDF
                sdf_path = molecules_dir / f"{safe_key}_{sub_key}.sdf"
                convert_mol_list_to_sdf(sub_value, sdf_path, smiles_key)
                entry_meta[sub_key] = {
                    "type": "mol_list",
                    "file": f"molecules/{safe_key}_{sub_key}.sdf",
                    "count": len(sub_value),
                }
            else:
                # Arrays -> npz
                collected = collect_arrays(sub_value, sub_key)
                for arr_key, arr_val in collected.items():
                    arrays_to_save[arr_key] = arr_val
                entry_meta[sub_key] = numpy_to_serializable(sub_value)

        if arrays_to_save:
            np.savez_compressed(arrays_dir / f"{safe_key}.npz", **arrays_to_save)
            entry_meta["__arrays_file__"] = f"arrays/{safe_key}.npz"

        metadata[smiles_key] = entry_meta

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


def convert_rdkit_samples_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert RDKit samples pickle to safe format.
    Structure: {smiles: {mol_unrelaxed: [Mol], mol_relaxed: [Mol]}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    molecules_dir = output_dir / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    for smiles_key, entry in data.items():
        safe_key = sanitize_key(smiles_key)
        entry_meta = {}

        for sub_key, sub_value in entry.items():
            if (
                isinstance(sub_value, list)
                and len(sub_value) > 0
                and isinstance(sub_value[0], Chem.Mol)
            ):
                sdf_path = molecules_dir / f"{safe_key}_{sub_key}.sdf"
                convert_mol_list_to_sdf(sub_value, sdf_path, smiles_key)
                entry_meta[sub_key] = {
                    "type": "mol_list",
                    "file": f"molecules/{safe_key}_{sub_key}.sdf",
                    "count": len(sub_value),
                }

        metadata[smiles_key] = entry_meta

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


def convert_puckerflow_samples_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert PuckerFlow samples pickle to safe format.
    Structure: {smiles: {ab: ndarray, mol: [Mol]}} or {smiles: {ab: ndarray, mol: [Mol], mol_relaxed: [Mol], ab_relaxed: ndarray}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    molecules_dir = output_dir / "molecules"
    molecules_dir.mkdir(exist_ok=True)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(exist_ok=True)

    for smiles_key, entry in data.items():
        safe_key = sanitize_key(smiles_key)
        entry_meta = {}
        arrays_to_save = {}

        for sub_key, sub_value in entry.items():
            if (
                isinstance(sub_value, list)
                and len(sub_value) > 0
                and isinstance(sub_value[0], Chem.Mol)
            ):
                sdf_path = molecules_dir / f"{safe_key}_{sub_key}.sdf"
                convert_mol_list_to_sdf(sub_value, sdf_path, smiles_key)
                entry_meta[sub_key] = {
                    "type": "mol_list",
                    "file": f"molecules/{safe_key}_{sub_key}.sdf",
                    "count": len(sub_value),
                }
            elif isinstance(sub_value, np.ndarray) or (
                HAS_TORCH and isinstance(sub_value, torch.Tensor)
            ):
                arrays_to_save[sub_key] = (
                    sub_value
                    if isinstance(sub_value, np.ndarray)
                    else sub_value.detach().cpu().numpy()
                )
                entry_meta[sub_key] = numpy_to_serializable(sub_value)

        if arrays_to_save:
            np.savez_compressed(arrays_dir / f"{safe_key}.npz", **arrays_to_save)
            entry_meta["__arrays_file__"] = f"arrays/{safe_key}.npz"

        metadata[smiles_key] = entry_meta

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


def convert_numpy_dict_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert numpy dict pickle to safe format.
    Structure: {smiles: ndarray}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    arrays = {}

    for smiles_key, arr in data.items():
        safe_key = sanitize_key(smiles_key)
        arrays[safe_key] = (
            arr if isinstance(arr, np.ndarray) else arr.detach().cpu().numpy()
        )
        metadata[smiles_key] = {
            "safe_key": safe_key,
            "dtype": str(arrays[safe_key].dtype),
            "shape": list(arrays[safe_key].shape),
        }

    np.savez_compressed(output_dir / "arrays.npz", **arrays)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


def convert_mol_dict_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert mol dict pickle to safe format.
    Structure: {smiles: [Mol, Mol, ...]}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    molecules_dir = output_dir / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    for smiles_key, mol_list in data.items():
        safe_key = sanitize_key(smiles_key)
        sdf_path = molecules_dir / f"{safe_key}.sdf"
        convert_mol_list_to_sdf(mol_list, sdf_path, smiles_key)
        metadata[smiles_key] = {
            "safe_key": safe_key,
            "file": f"molecules/{safe_key}.sdf",
            "count": len(mol_list),
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


def convert_pyg_dataset_to_safe(pickle_path: Path, output_dir: Path) -> None:
    """
    Convert PyTorch Geometric dataset pickle to safe format.
    Structure: {smiles: [Data, Data, ...]}
    Data can contain tensors, numpy arrays, and RDKit molecules.
    """
    if not HAS_PYG:
        print(f"  SKIP: torch_geometric not installed, cannot convert {pickle_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    metadata = {}
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(exist_ok=True)
    molecules_dir = output_dir / "molecules"
    molecules_dir.mkdir(exist_ok=True)

    for smiles_key, data_list in data.items():
        safe_key = sanitize_key(smiles_key)
        entry_meta = {"count": len(data_list), "data_items": []}
        all_arrays = {}
        mol_list = []  # Collect all molecules from this SMILES

        for i, pyg_data in enumerate(data_list):
            item_meta = {}
            for attr in pyg_data.keys():
                value = pyg_data[attr]
                arr_key = f"{i}_{attr}"
                if isinstance(value, Chem.Mol):
                    # Store molecule in list, save index
                    mol_idx = len(mol_list)
                    mol_list.append(value)
                    item_meta[attr] = {
                        "__mol__": True,
                        "mol_idx": mol_idx,
                    }
                elif HAS_TORCH and isinstance(value, torch.Tensor):
                    all_arrays[arr_key] = value.detach().cpu().numpy()
                    item_meta[attr] = {
                        "__torch__": True,
                        "dtype": str(all_arrays[arr_key].dtype),
                        "shape": list(all_arrays[arr_key].shape),
                        "arr_key": arr_key,
                    }
                elif isinstance(value, np.ndarray):
                    all_arrays[arr_key] = value
                    item_meta[attr] = {
                        "__numpy__": True,
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "arr_key": arr_key,
                    }
                else:
                    item_meta[attr] = value
            entry_meta["data_items"].append(item_meta)

        np.savez_compressed(arrays_dir / f"{safe_key}.npz", **all_arrays)

        # Save molecules if any
        if mol_list:
            sdf_path = molecules_dir / f"{safe_key}.sdf"
            convert_mol_list_to_sdf(mol_list, sdf_path, smiles_key)
            entry_meta["mol_file"] = f"molecules/{safe_key}.sdf"
            entry_meta["mol_count"] = len(mol_list)

        metadata[smiles_key] = entry_meta

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Converted {len(data)} entries to {output_dir}")


# ============================================================================
# Type-specific converters: to_pickle
# ============================================================================


def convert_safe_to_samples_with_config(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to MCF/GeoDiff samples pickle."""
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = {}
    for smiles_key, entry_meta in metadata.items():
        entry = {}
        safe_key = sanitize_key(smiles_key)

        # Load arrays if present
        arrays = {}
        if "__arrays_file__" in entry_meta:
            npz_path = input_dir / entry_meta["__arrays_file__"]
            with np.load(npz_path) as npz:
                arrays = dict(npz)

        for sub_key, sub_value in entry_meta.items():
            if sub_key == "__arrays_file__":
                continue
            elif isinstance(sub_value, dict) and sub_value.get("type") == "mol_list":
                sdf_path = input_dir / sub_value["file"]
                entry[sub_key] = load_mol_list_from_sdf(sdf_path)
            elif isinstance(sub_value, dict) and (
                sub_value.get("__numpy__") or sub_value.get("__torch__")
            ):
                # Reconstruct array from npz
                if sub_key in arrays:
                    if sub_value.get("__torch__"):
                        dtype_str = sub_value.get("dtype", "float32")
                        dtype_map = {
                            "float32": torch.float32,
                            "float64": torch.float64,
                            "int32": torch.int32,
                            "int64": torch.int64,
                        }
                        entry[sub_key] = torch.tensor(
                            arrays[sub_key],
                            dtype=dtype_map.get(dtype_str, torch.float32),
                        )
                    else:
                        entry[sub_key] = arrays[sub_key]
            elif isinstance(sub_value, list):
                # Nested list with arrays
                reconstructed = []
                for i, item in enumerate(sub_value):
                    arr_key = f"{sub_key}[{i}]"
                    if arr_key in arrays:
                        reconstructed.append(arrays[arr_key].tolist())
                    elif isinstance(item, dict) and (
                        item.get("__numpy__") or item.get("__torch__")
                    ):
                        arr_key = f"{sub_key}[{i}]"
                        if arr_key in arrays:
                            reconstructed.append(arrays[arr_key].tolist())
                        else:
                            reconstructed.append(item)
                    else:
                        reconstructed.append(item)
                entry[sub_key] = reconstructed
            else:
                entry[sub_key] = sub_value

        data[smiles_key] = entry

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


def convert_safe_to_rdkit_samples(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to RDKit samples pickle."""
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = {}
    for smiles_key, entry_meta in metadata.items():
        entry = {}
        for sub_key, sub_value in entry_meta.items():
            if isinstance(sub_value, dict) and sub_value.get("type") == "mol_list":
                sdf_path = input_dir / sub_value["file"]
                entry[sub_key] = load_mol_list_from_sdf(sdf_path)
        data[smiles_key] = entry

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


def convert_safe_to_puckerflow_samples(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to PuckerFlow samples pickle."""
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = {}
    for smiles_key, entry_meta in metadata.items():
        entry = {}
        safe_key = sanitize_key(smiles_key)

        # Load arrays if present
        arrays = {}
        if "__arrays_file__" in entry_meta:
            npz_path = input_dir / entry_meta["__arrays_file__"]
            with np.load(npz_path) as npz:
                arrays = dict(npz)

        for sub_key, sub_value in entry_meta.items():
            if sub_key == "__arrays_file__":
                continue
            elif isinstance(sub_value, dict) and sub_value.get("type") == "mol_list":
                sdf_path = input_dir / sub_value["file"]
                entry[sub_key] = load_mol_list_from_sdf(sdf_path)
            elif isinstance(sub_value, dict) and sub_value.get("__numpy__"):
                entry[sub_key] = arrays[sub_key]
            elif isinstance(sub_value, dict) and sub_value.get("__torch__"):
                dtype_str = sub_value.get("dtype", "float32")
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int32": torch.int32,
                    "int64": torch.int64,
                }
                entry[sub_key] = torch.tensor(
                    arrays[sub_key], dtype=dtype_map.get(dtype_str, torch.float32)
                )

        data[smiles_key] = entry

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


def convert_safe_to_numpy_dict(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to numpy dict pickle."""
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with np.load(input_dir / "arrays.npz") as npz:
        arrays = dict(npz)

    data = {}
    for smiles_key, entry_meta in metadata.items():
        safe_key = entry_meta["safe_key"]
        data[smiles_key] = arrays[safe_key]

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


def convert_safe_to_mol_dict(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to mol dict pickle."""
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = {}
    for smiles_key, entry_meta in metadata.items():
        sdf_path = input_dir / entry_meta["file"]
        data[smiles_key] = load_mol_list_from_sdf(sdf_path)

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


def convert_safe_to_pyg_dataset(input_dir: Path, output_pickle: Path) -> None:
    """Convert safe format back to PyTorch Geometric dataset pickle."""
    if not HAS_PYG:
        print(f"  SKIP: torch_geometric not installed, cannot convert {input_dir}")
        return

    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    data = {}
    arrays_dir = input_dir / "arrays"
    molecules_dir = input_dir / "molecules"

    for smiles_key, entry_meta in metadata.items():
        safe_key = sanitize_key(smiles_key)

        with np.load(arrays_dir / f"{safe_key}.npz") as npz:
            arrays = dict(npz)

        # Load molecules if present
        mol_list = None
        if "mol_file" in entry_meta:
            sdf_path = input_dir / entry_meta["mol_file"]
            mol_list = load_mol_list_from_sdf(sdf_path)

        data_list = []
        for i, item_meta in enumerate(entry_meta["data_items"]):
            kwargs = {}
            for attr, attr_meta in item_meta.items():
                if isinstance(attr_meta, dict) and attr_meta.get("__mol__"):
                    mol_idx = attr_meta["mol_idx"]
                    kwargs[attr] = mol_list[mol_idx]
                elif isinstance(attr_meta, dict) and attr_meta.get("__torch__"):
                    arr_key = attr_meta["arr_key"]
                    dtype_str = attr_meta.get("dtype", "float32")
                    dtype_map = {
                        "float32": torch.float32,
                        "float64": torch.float64,
                        "int32": torch.int32,
                        "int64": torch.int64,
                        "bool": torch.bool,
                    }
                    kwargs[attr] = torch.tensor(
                        arrays[arr_key], dtype=dtype_map.get(dtype_str, torch.float32)
                    )
                elif isinstance(attr_meta, dict) and attr_meta.get("__numpy__"):
                    arr_key = attr_meta["arr_key"]
                    kwargs[attr] = arrays[arr_key]
                else:
                    kwargs[attr] = attr_meta
            data_list.append(PyGData(**kwargs))

        data[smiles_key] = data_list

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Converted {len(data)} entries to {output_pickle}")


# ============================================================================
# Main conversion dispatcher
# ============================================================================

CONVERTERS_TO_SAFE = {
    "samples_with_config": convert_samples_with_config_to_safe,
    "rdkit_samples": convert_rdkit_samples_to_safe,
    "puckerflow_samples": convert_puckerflow_samples_to_safe,
    "numpy_dict": convert_numpy_dict_to_safe,
    "mol_dict": convert_mol_dict_to_safe,
    "pyg_dataset": convert_pyg_dataset_to_safe,
}

CONVERTERS_TO_PICKLE = {
    "samples_with_config": convert_safe_to_samples_with_config,
    "rdkit_samples": convert_safe_to_rdkit_samples,
    "puckerflow_samples": convert_safe_to_puckerflow_samples,
    "numpy_dict": convert_safe_to_numpy_dict,
    "mol_dict": convert_safe_to_mol_dict,
    "pyg_dataset": convert_safe_to_pyg_dataset,
}


def get_all_pickle_paths(base_dir: Path) -> List[Dict]:
    """Get all pickle files and their configurations."""
    files = []

    # Seed-based files
    for config_name, config in PICKLE_CONFIGS.items():
        for seed in SEEDS:
            pickle_path = base_dir / config["pattern"].format(seed=seed)
            safe_dir = base_dir / config["safe_dir"].format(seed=seed)
            if pickle_path.exists():
                files.append(
                    {
                        "name": f"{config_name}_{seed}",
                        "pickle_path": pickle_path,
                        "safe_dir": safe_dir,
                        "type": config["type"],
                    }
                )

    # Non-seed files
    for config_name, config in NON_SEED_PICKLE_CONFIGS.items():
        pickle_path = base_dir / config["pattern"]
        safe_dir = base_dir / config["safe_dir"]
        if pickle_path.exists():
            files.append(
                {
                    "name": config_name,
                    "pickle_path": pickle_path,
                    "safe_dir": safe_dir,
                    "type": config["type"],
                }
            )

    return files


def convert_all_to_safe(base_dir: Path) -> None:
    """Convert all pickle files to safe format."""
    files = get_all_pickle_paths(base_dir)

    print(f"Found {len(files)} pickle files to convert")
    print("=" * 80)

    for file_info in files:
        print(f"\nConverting: {file_info['name']}")
        print(f"  From: {file_info['pickle_path']}")
        print(f"  To: {file_info['safe_dir']}")

        try:
            converter = CONVERTERS_TO_SAFE[file_info["type"]]
            converter(file_info["pickle_path"], file_info["safe_dir"])
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print("Conversion complete!")


def convert_all_to_pickle(base_dir: Path) -> None:
    """Convert all safe format directories back to pickle files."""
    files = []

    # Seed-based files
    for config_name, config in PICKLE_CONFIGS.items():
        for seed in SEEDS:
            safe_dir = base_dir / config["safe_dir"].format(seed=seed)
            pickle_path = base_dir / config["pattern"].format(seed=seed)
            if safe_dir.exists() and (safe_dir / "metadata.json").exists():
                files.append(
                    {
                        "name": f"{config_name}_{seed}",
                        "pickle_path": pickle_path,
                        "safe_dir": safe_dir,
                        "type": config["type"],
                    }
                )

    # Non-seed files
    for config_name, config in NON_SEED_PICKLE_CONFIGS.items():
        safe_dir = base_dir / config["safe_dir"]
        pickle_path = base_dir / config["pattern"]
        if safe_dir.exists() and (safe_dir / "metadata.json").exists():
            files.append(
                {
                    "name": config_name,
                    "pickle_path": pickle_path,
                    "safe_dir": safe_dir,
                    "type": config["type"],
                }
            )

    print(f"Found {len(files)} safe directories to convert")
    print("=" * 80)

    for file_info in files:
        print(f"\nConverting: {file_info['name']}")
        print(f"  From: {file_info['safe_dir']}")
        print(f"  To: {file_info['pickle_path']}")

        try:
            converter = CONVERTERS_TO_PICKLE[file_info["type"]]
            converter(file_info["safe_dir"], file_info["pickle_path"])
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print("Conversion complete!")


# ============================================================================
# Zip/Unzip functions for compact distribution
# ============================================================================


def zip_all_safe_dirs(base_dir: Path) -> None:
    """Compress all safe directories into .tar.gz archives for distribution."""
    import shutil
    import tarfile

    safe_dirs = []

    # Seed-based files
    for config_name, config in PICKLE_CONFIGS.items():
        for seed in SEEDS:
            safe_dir = base_dir / config["safe_dir"].format(seed=seed)
            if safe_dir.exists():
                safe_dirs.append(safe_dir)

    # Non-seed files
    for config_name, config in NON_SEED_PICKLE_CONFIGS.items():
        safe_dir = base_dir / config["safe_dir"]
        if safe_dir.exists():
            safe_dirs.append(safe_dir)

    print(f"Found {len(safe_dirs)} safe directories to compress")
    print("=" * 80)

    total_original = 0
    total_compressed = 0

    for safe_dir in safe_dirs:
        archive_path = safe_dir.parent / f"{safe_dir.name}.tar.gz"
        print(f"\nCompressing: {safe_dir.relative_to(base_dir)}")

        # Calculate original size
        original_size = sum(
            f.stat().st_size for f in safe_dir.rglob("*") if f.is_file()
        )
        total_original += original_size

        # Create tar.gz archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(safe_dir, arcname=safe_dir.name)

        compressed_size = archive_path.stat().st_size
        total_compressed += compressed_size

        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        print(
            f"  {original_size / 1024:.1f} KB -> {compressed_size / 1024:.1f} KB ({ratio:.1f}% reduction)"
        )

        # Remove original directory
        shutil.rmtree(safe_dir)

    print("\n" + "=" * 80)
    ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
    print(
        f"Total: {total_original / 1024 / 1024:.1f} MB -> {total_compressed / 1024 / 1024:.1f} MB ({ratio:.1f}% reduction)"
    )
    print(
        "Compression complete! Safe directories have been replaced with .tar.gz archives."
    )


def unzip_all_safe_archives(base_dir: Path) -> None:
    """Extract all .tar.gz archives back to safe directories."""
    import tarfile

    archives = []

    # Find all *.tar.gz archives
    for pattern in ["*.tar.gz"]:
        archives.extend(base_dir.rglob(pattern))

    if not archives:
        print("No .tar.gz archives found. Checking if directories already exist...")
        # Check if safe directories already exist (maybe already extracted)
        safe_dirs_exist = False
        for config_name, config in PICKLE_CONFIGS.items():
            for seed in SEEDS:
                safe_dir = base_dir / config["safe_dir"].format(seed=seed)
                if safe_dir.exists():
                    safe_dirs_exist = True
                    break

        if safe_dirs_exist:
            print("Safe directories already exist. No extraction needed.")
        else:
            print("ERROR: No archives or safe directories found!")
            sys.exit(1)
        return

    print(f"Found {len(archives)} archives to extract")
    print("=" * 80)

    for archive_path in sorted(archives):
        print(f"\nExtracting: {archive_path.relative_to(base_dir)}")

        # Extract to parent directory
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=archive_path.parent)

        # Remove archive after extraction
        archive_path.unlink()
        print(f"  Extracted and removed archive")

    print("\n" + "=" * 80)
    print("Extraction complete!")


# ============================================================================
# Verification functions
# ============================================================================


def compare_molecules(mol1: Chem.Mol, mol2: Chem.Mol, atol: float = 1e-4) -> bool:
    """Compare two RDKit molecules for equivalence."""
    if mol1 is None and mol2 is None:
        return True
    if mol1 is None or mol2 is None:
        return False

    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False
    if mol1.GetNumConformers() != mol2.GetNumConformers():
        return False

    # Compare SMILES
    smi1 = Chem.MolToSmiles(mol1)
    smi2 = Chem.MolToSmiles(mol2)
    if smi1 != smi2:
        return False

    # Compare conformer coordinates
    for i in range(mol1.GetNumConformers()):
        conf1 = mol1.GetConformer(i)
        conf2 = mol2.GetConformer(i)
        for j in range(mol1.GetNumAtoms()):
            pos1 = conf1.GetAtomPosition(j)
            pos2 = conf2.GetAtomPosition(j)
            if not np.allclose(
                [pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z], atol=atol
            ):
                return False

    return True


def verify_conversion(original_pkl: Path, restored_pkl: Path) -> bool:
    """Verify that two pickle files contain equivalent data."""
    print(f"Loading original: {original_pkl}")
    with open(original_pkl, "rb") as f:
        original = pickle.load(f)

    print(f"Loading restored: {restored_pkl}")
    with open(restored_pkl, "rb") as f:
        restored = pickle.load(f)

    print("\nVerification:")
    print("=" * 80)

    if type(original) != type(restored):
        print(f"FAIL: Top-level type mismatch: {type(original)} vs {type(restored)}")
        return False

    if not isinstance(original, dict):
        print(f"FAIL: Expected dict, got {type(original)}")
        return False

    orig_keys = set(original.keys())
    rest_keys = set(restored.keys())

    if orig_keys != rest_keys:
        print(f"FAIL: Keys mismatch")
        print(f"  Missing in restored: {orig_keys - rest_keys}")
        print(f"  Extra in restored: {rest_keys - orig_keys}")
        return False

    print(f"✓ Number of top-level keys: {len(orig_keys)}")

    total_issues = 0
    mol_count = 0
    array_count = 0

    for key in sorted(orig_keys):
        orig_entry = original[key]
        rest_entry = restored[key]

        # Handle list of molecules/Data objects
        if isinstance(orig_entry, list):
            if len(orig_entry) != len(rest_entry):
                print(
                    f"FAIL: {key}: List length mismatch {len(orig_entry)} vs {len(rest_entry)}"
                )
                total_issues += 1
                continue

            for i, (o, r) in enumerate(zip(orig_entry, rest_entry)):
                if isinstance(o, Chem.Mol):
                    if not compare_molecules(o, r):
                        print(f"FAIL: {key}[{i}]: Molecules differ")
                        total_issues += 1
                    else:
                        mol_count += 1
                elif HAS_PYG and isinstance(o, PyGData):
                    # Compare PyG Data objects
                    for attr in o.keys():
                        if attr not in r.keys():
                            print(f"FAIL: {key}[{i}]: Missing attribute {attr}")
                            total_issues += 1
                        else:
                            o_val = o[attr]
                            r_val = r[attr]
                            if isinstance(o_val, Chem.Mol):
                                if not compare_molecules(o_val, r_val):
                                    print(f"FAIL: {key}[{i}].{attr}: Molecules differ")
                                    total_issues += 1
                                else:
                                    mol_count += 1
                            elif HAS_TORCH and isinstance(o_val, torch.Tensor):
                                if not torch.allclose(
                                    o_val.float(), r_val.float(), atol=1e-5
                                ):
                                    print(
                                        f"FAIL: {key}[{i}].{attr}: Tensor values differ"
                                    )
                                    total_issues += 1
                                else:
                                    array_count += 1
                            elif isinstance(o_val, np.ndarray):
                                if not np.allclose(o_val, r_val, atol=1e-5):
                                    print(
                                        f"FAIL: {key}[{i}].{attr}: Array values differ"
                                    )
                                    total_issues += 1
                                else:
                                    array_count += 1
            continue

        # Handle dict entries
        if isinstance(orig_entry, dict):
            for sub_key in orig_entry.keys():
                if sub_key not in rest_entry:
                    print(f"FAIL: {key}: Missing sub-key {sub_key}")
                    total_issues += 1
                    continue

                orig_value = orig_entry[sub_key]
                rest_value = rest_entry[sub_key]

                if (
                    isinstance(orig_value, list)
                    and len(orig_value) > 0
                    and isinstance(orig_value[0], Chem.Mol)
                ):
                    if len(orig_value) != len(rest_value):
                        print(f"FAIL: {key}/{sub_key}: Molecule list length mismatch")
                        total_issues += 1
                    else:
                        for i, (m1, m2) in enumerate(zip(orig_value, rest_value)):
                            if not compare_molecules(m1, m2):
                                print(f"FAIL: {key}/{sub_key}[{i}]: Molecules differ")
                                total_issues += 1
                            else:
                                mol_count += 1
                elif isinstance(orig_value, np.ndarray):
                    if not np.allclose(orig_value, rest_value, atol=1e-5):
                        print(f"FAIL: {key}/{sub_key}: Array values differ")
                        total_issues += 1
                    else:
                        array_count += 1
                elif HAS_TORCH and isinstance(orig_value, torch.Tensor):
                    if not torch.allclose(
                        orig_value.float(), rest_value.float(), atol=1e-5
                    ):
                        print(f"FAIL: {key}/{sub_key}: Tensor values differ")
                        total_issues += 1
                    else:
                        array_count += 1

        # Handle numpy array entries
        elif isinstance(orig_entry, np.ndarray):
            if not np.allclose(orig_entry, rest_entry, atol=1e-5):
                print(f"FAIL: {key}: Array values differ")
                total_issues += 1
            else:
                array_count += 1

    print(f"\n✓ Verified {mol_count} molecules")
    print(f"✓ Verified {array_count} arrays/tensors")

    if total_issues == 0:
        print("\n" + "=" * 80)
        print("SUCCESS: All data matches!")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print(f"FAILURE: Found {total_issues} issues")
        print("=" * 80)
        return False


# ============================================================================
# Main entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert pickle files to safe format for publishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all pickles to safe format
  python convert_pickles.py to_safe

  # Compress safe directories into .tar.gz archives (for GitHub upload)
  python convert_pickles.py zip

  # Extract .tar.gz archives back to safe directories
  python convert_pickles.py unzip

  # Convert safe format back to pickles
  python convert_pickles.py to_pickle

  # Verify a conversion
  python convert_pickles.py verify original.pkl restored.pkl

  # Convert a single file
  python convert_pickles.py to_safe --file input.pkl --output output_dir --type samples_with_config
  python convert_pickles.py to_pickle --input input_dir --output output.pkl --type samples_with_config

Available types:
  - samples_with_config: MCF/GeoDiff samples with config dict
  - rdkit_samples: RDKit conformer samples
  - puckerflow_samples: PuckerFlow conformer samples
  - numpy_dict: Dictionary of numpy arrays
  - mol_dict: Dictionary of molecule lists
  - pyg_dataset: PyTorch Geometric dataset
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # to_safe command
    to_safe_parser = subparsers.add_parser(
        "to_safe", help="Convert pickles to safe format"
    )
    to_safe_parser.add_argument(
        "--file", type=str, help="Single pickle file to convert"
    )
    to_safe_parser.add_argument("--output", type=str, help="Output directory")
    to_safe_parser.add_argument(
        "--type", type=str, choices=list(CONVERTERS_TO_SAFE.keys()), help="Data type"
    )
    to_safe_parser.add_argument(
        "--base-dir", type=str, default=".", help="Base directory for all files"
    )

    # to_pickle command
    to_pickle_parser = subparsers.add_parser(
        "to_pickle", help="Convert safe format to pickles"
    )
    to_pickle_parser.add_argument("--input", type=str, help="Input safe directory")
    to_pickle_parser.add_argument("--output", type=str, help="Output pickle file")
    to_pickle_parser.add_argument(
        "--type", type=str, choices=list(CONVERTERS_TO_PICKLE.keys()), help="Data type"
    )
    to_pickle_parser.add_argument(
        "--base-dir", type=str, default=".", help="Base directory for all files"
    )

    # zip command
    zip_parser = subparsers.add_parser(
        "zip", help="Compress safe directories into .tar.gz archives"
    )
    zip_parser.add_argument(
        "--base-dir", type=str, default=".", help="Base directory for all files"
    )

    # unzip command
    unzip_parser = subparsers.add_parser(
        "unzip", help="Extract .tar.gz archives to safe directories"
    )
    unzip_parser.add_argument(
        "--base-dir", type=str, default=".", help="Base directory for all files"
    )

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify conversion")
    verify_parser.add_argument("original", type=str, help="Original pickle file")
    verify_parser.add_argument("restored", type=str, help="Restored pickle file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "to_safe":
        if args.file:
            if not args.output or not args.type:
                print("Error: --output and --type are required when using --file")
                sys.exit(1)
            converter = CONVERTERS_TO_SAFE[args.type]
            converter(Path(args.file), Path(args.output))
        else:
            convert_all_to_safe(Path(args.base_dir))

    elif args.command == "to_pickle":
        if args.input:
            if not args.output or not args.type:
                print("Error: --output and --type are required when using --input")
                sys.exit(1)
            converter = CONVERTERS_TO_PICKLE[args.type]
            converter(Path(args.input), Path(args.output))
        else:
            convert_all_to_pickle(Path(args.base_dir))

    elif args.command == "zip":
        zip_all_safe_dirs(Path(args.base_dir))

    elif args.command == "unzip":
        unzip_all_safe_archives(Path(args.base_dir))

    elif args.command == "verify":
        success = verify_conversion(Path(args.original), Path(args.restored))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
