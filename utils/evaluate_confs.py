"""
Evaluation script for conformer generation models.

Calculates RMSD, RMSE (from puckering coordinates), and 'ab_diff' metrics
to compare generated conformers (relaxed and unrelaxed) against a
ground truth dataset.

Supports multiprocessing for parallel metric calculation.
"""

from datetime import datetime
import json
import logging
import os
import pathlib
import pickle
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from utils.coordinate_transforms import ab_to_z, mol_to_abs, mol_to_z
from utils.utils import merge_dicts

LOGGER = logging.getLogger(__name__)
true_confs_dict: Dict[str, Any] = {}
sample_confs_dict: Dict[str, Any] = {}

def init_worker(
    true_confs_path: str,
    sample_confs_path: str,
) -> None:
    global true_confs_dict, sample_confs_dict

    with open(true_confs_path, "rb") as f:
        true_confs_dict = pickle.load(f)

    with open(sample_confs_path, "rb") as f:
        sample_confs_dict = pickle.load(f)
    LOGGER.debug(
        "Worker %s init: true=%d sample=%d",
        os.getpid(),
        len(true_confs_dict),
        len(sample_confs_dict),
    )


def calc_min_rmse(sample_z: torch.Tensor, true_z_stack: torch.Tensor) -> float:
    """
    Calculate the minimum RMSE between a sample's z-coordinates
    and a stack of true z-coordinates (for all renumberings).

    Args:
        sample_z: 1D Tensor of z-coordinates for the sample conformer.
        true_z_stack: 2D Tensor (num_renumberings, N) of z-coordinates
                      for the ground truth conformer.

    Returns:
        The minimum RMSE value.
    """
    mse_array = torch.mean((true_z_stack - sample_z[None, :]) ** 2, dim=1)
    rmse_array = torch.sqrt(mse_array)
    return float(torch.min(rmse_array))


def create_cyclic_bonds_from_xyz_order(mol: Chem.Mol) -> Chem.Mol:
    """
    Create bonds between consecutive atoms in XYZ order to form a cycle.

    This is a fallback for when MolFromXYZBlock fails to infer connectivity.

    Args:
        mol: An RDKit Mol object, likely without bonds.

    Returns:
        A new RDKit Mol object with cyclic bonds, or the original
        mol if sanitization fails.
    """
    rwmol = Chem.RWMol(mol)

    # Clear existing bonds
    bonds_to_remove = []
    for bond in Chem.Mol.GetBonds(rwmol):
        bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    for begin_idx, end_idx in bonds_to_remove:
        rwmol.RemoveBond(begin_idx, end_idx)

    num_atoms = rwmol.GetNumAtoms()

    # Create bonds between consecutive atoms
    for i in range(num_atoms):
        next_i = (i + 1) % num_atoms  # Wrap around to create cycle
        rwmol.AddBond(i, next_i, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rwmol)
        result_mol = rwmol.GetMol()
        logging.info(f"Created cyclic structure: {Chem.MolToSmiles(result_mol)}")
        return result_mol
    except Exception as e:
        logging.error(f"Failed to sanitize cyclic molecule: {e}")
        return mol  # Return original mol as fallback


def get_rdkit_mol_with_smiles_and_xyz_block(smiles: str, xyz_block: str) -> Chem.Mol:
    """
    Create an RDKit Mol object from an XYZ block and assign bond orders
    using a SMILES template.

    Args:
        smiles: The reference SMILES string.
        xyz_block: The XYZ coordinate block as a string.

    Returns:
        An RDKit Mol object with 3D coordinates and assigned bond orders.

    Raises:
        ValueError: If bond order assignment fails.
    """
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError(f"Could not parse XYZ block for SMILES {smiles}")

    ref_mol = Chem.MolFromSmiles(smiles)
    if ref_mol is None:
        raise ValueError(f"Could not parse reference SMILES: {smiles}")

    try:
        # Fallback to create bonds in a cycle
        mol = create_cyclic_bonds_from_xyz_order(mol)
        mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol)
        return cast(Chem.Mol, mol)
    except Exception as e:
        logging.info(f"Getting mol failed for {smiles}")
        logging.info(f"XYZ block:\n{xyz_block}")
        logging.info(f"{Chem.MolToSmiles(mol)} vs {smiles}")
        raise ValueError(
            f"Could not assign bond orders from template for SMILES {smiles}: {e}"
        )


def get_rdkit_mol_with_smiles_and_mol(smiles: str, mol: Chem.Mol) -> Chem.Mol:
    """
    Assign bond orders to a Mol object using a SMILES template.

    This function acts as a wrapper, converting the Mol object
    to an XYZ block first.

    Args:
        smiles: The reference SMILES string.
        mol: The RDKit Mol object (with 3D coordinates but possibly
             incorrect bond orders).

    Returns:
        A new RDKit Mol object with corrected bond orders.
    """
    try:
        xyz_block = Chem.MolToXYZBlock(mol)
        return get_rdkit_mol_with_smiles_and_xyz_block(smiles, xyz_block)
    except Exception as e:
        logging.info(f"Getting mol failed for {smiles}")
        logging.info(f"{Chem.MolToSmiles(mol)} vs {smiles}")
        raise ValueError(
            f"Could not assign bond orders from template for SMILES {smiles}: {e}"
        )


def get_rdkit_mol_with_xyz_file(xyz_file: str) -> Chem.Mol:
    """
    Load a molecule from an XYZ file, inferring SMILES from the filename.

    Args:
        xyz_file: The path to the .xyz file.

    Returns:
        An RDKit Mol object.

    Raises:
        ValueError: If the SMILES from the filename does not match
                    the SMILES of the resulting molecule.
    """
    smiles = xyz_file.split("/")[-1].split("_")[0]

    with open(xyz_file, "r") as f:
        xyz_block = f.read()

    mol = get_rdkit_mol_with_smiles_and_xyz_block(smiles, xyz_block)

    if smiles != Chem.MolToSmiles(mol):
        raise ValueError(f"SMILES mismatch: {smiles} vs {Chem.MolToSmiles(mol)}")
    return mol


def calc_performance_stats(
    difference_array: np.ndarray, thresholds: List[float]
) -> Dict[str, float]:
    """
    Calculate coverage (precision/recall) and AMR (precision/recall)
    from a difference matrix.

    Args:
        difference_array: 2D NumPy array (n_true, n_sample) of
                          differences (e.g., RMSD values).
        thresholds: A list of thresholds to compute coverage at.

    Returns:
        A dictionary of statistics.
    """
    stats: Dict[str, float] = {}
    for threshold in thresholds:
        key_suffix = f"_{threshold}"
        # Recall: For each true conf, is there at least one sample close enough?
        stats[f"coverage_recall{key_suffix}"] = float(
            np.mean(difference_array.min(axis=1) < threshold) * 100
        )
        # Precision: For each sample conf, is it close to at least one true conf?
        stats[f"coverage_precision{key_suffix}"] = float(
            np.mean(difference_array.min(axis=0) < threshold) * 100
        )

    # AMR (Average Minimum RMSD)
    stats["amr_recall"] = difference_array.min(axis=1).mean()
    stats["amr_precision"] = difference_array.min(axis=0).mean()

    return stats


def get_stats_and_print(
    label: str, thresholds: List[float], results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate and print mean/median statistics for a given metric.

    Args:
        label: The metric label (e.g., 'rmsd_relaxed').
        thresholds: A list of thresholds.
        results: The main results dictionary containing difference arrays.

    Returns:
        A dictionary of aggregated statistics.
    """
    print(f"\n--- Statistics for {label} ---")
    stat_dicts: List[Dict[str, float]] = []
    result: Dict[str, float] = {}

    for res in results.values():
        if res.get("n_sample", 0) == 0:
            continue
        stat_dict = calc_performance_stats(res[label], thresholds)
        stat_dicts.append(stat_dict)

    if not stat_dicts:
        print(f"No results found for {label}.")
        return result

    all_stats = merge_dicts(*stat_dicts, nested=False)

    # Print threshold-specific metrics
    for threshold in thresholds:
        key_suffix = f"_{threshold}"
        print(f"\nResults for threshold = {threshold}:")

        for key_base in ["coverage_recall", "coverage_precision"]:
            key = f"{key_base}{key_suffix}"
            mean_val = np.nanmean(all_stats[key])
            median_val = np.median(all_stats[key])
            result[f"{label}_mean_{key}"] = float(mean_val)
            result[f"{label}_median_{key}"] = float(median_val)

            print(
                f'  {key_base.replace("_", " ").title()}: Mean = {mean_val:.3f}, Median = {median_val:.3f}'
            )

    # Print threshold-independent metrics
    print("\nThreshold-independent metrics:")
    for key in ["amr_recall", "amr_precision"]:
        mean_val = np.nanmean(all_stats[key])
        median_val = np.median(all_stats[key])
        result[f"{label}_mean_{key}"] = float(mean_val)
        result[f"{label}_median_{key}"] = float(median_val)
        print(
            f'  {key.replace("_", " ").title()}: Mean = {mean_val:.3f}, Median = {median_val:.3f}'
        )

    return result


def worker_fn_ab(
    job: Tuple[str, int],
) -> Tuple[
    str,
    int,
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    Multiprocessing worker for `evaluate_with_ab`.
    Calculates RMSD, ab_diff, and RMSE for one true conformer
    against all sample conformers.
    """
    smi, i_true = job
    tc = true_confs_dict[smi]["mol"][i_true]

    # RMSD calculation
    rmsds_unrelaxed = [
        AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))  # type: ignore
        for mc in sample_confs_dict[smi]["mol_unrelaxed"]
    ]
    rmsds_relaxed = [
        AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))  # type: ignore
        for mc in sample_confs_dict[smi]["mol_relaxed"]
    ]

    # ab_diff calculation
    tc_ab = np.array(true_confs_dict[smi]["ab"][i_true])
    ab_diffs_unrelaxed = [
        float(np.linalg.norm(mc_ab - tc_ab))
        for mc_ab in sample_confs_dict[smi]["ab_unrelaxed"]
    ]
    ab_diffs_relaxed = [
        float(np.linalg.norm(mc_ab - tc_ab))
        for mc_ab in sample_confs_dict[smi]["ab_relaxed"]
    ]

    # RMSE calculation
    true_z_stack = true_confs_dict[smi]["z_stack"][i_true]
    rmses_unrelaxed = [
        calc_min_rmse(mol_to_z(mc), true_z_stack)
        for mc in sample_confs_dict[smi]["mol_unrelaxed"]
    ]
    rmses_relaxed = [
        calc_min_rmse(mol_to_z(mc), true_z_stack)
        for mc in sample_confs_dict[smi]["mol_relaxed"]
    ]
    return (
        smi,
        i_true,
        rmsds_unrelaxed,
        rmsds_relaxed,
        ab_diffs_unrelaxed,
        ab_diffs_relaxed,
        rmses_unrelaxed,
        rmses_relaxed,
    )


def evaluate_with_ab(
    config: Dict[str, Any], epoch: Union[int, str] = 0
) -> Dict[str, float]:
    """
    Main evaluation function when 'ab' (puckering) coordinates are available.

    1. Loads true conformers, original molecules, and samples.
    2. Performs MMFF relaxation on generated samples.
    3. Calculates 'ab' coordinates for all samples.
    4. Farms out metric calculation (RMSD, ab_diff, RMSE) to worker processes.
    5. Aggregates results and prints statistics.

    Args:
        config: The main configuration dictionary.
        epoch: The epoch number (or folder name) to load samples from.

    Returns:
        A dictionary of aggregated statistics.
    """
    with open(config["all_confs_path"], "rb") as f:
        all_confs_dict = pickle.load(f)
    with open(config["mol_dict_path"], "rb") as f:
        orig_mol_dict = pickle.load(f)
    with open(f"{config['subfolder']}/{config['sample_subfolder']}.pkl", "rb") as f:
        sample_confs_dict = pickle.load(f)

    mode = "test"
    with open(config["split_json_path"], "r") as f:
        test_smiles = json.load(f)[mode]

    # 1. Relax generated conformers
    LOGGER.info("Optimizing generated conformers...")
    for smi, smi_dict in tqdm(sample_confs_dict.items(), desc="Optimizing conformers"):
        for mol_unrelaxed in smi_dict["mol"]:
            mol_relaxed = Chem.Mol(mol_unrelaxed)
            try:
                AllChem.MMFFOptimizeMolecule(mol_relaxed)  # type: ignore
            except Exception:
                LOGGER.error("MMFF optimization failed for %s, trying UFF", smi)
                try:
                    AllChem.UFFOptimizeMolecule(mol_relaxed)  # type: ignore
                except Exception as e:
                    LOGGER.error("UFF optimization failed for %s: %s", smi, e)

            sample_confs_dict[smi].setdefault("mol_relaxed", []).append(mol_relaxed)

    # 2. Calculate 'ab' coordinates for all samples
    LOGGER.info("Calculating 'ab' coordinates for samples...")
    for smi, smi_sample_dict in sample_confs_dict.items():
        sample_confs_dict[smi] = {
            "mol_relaxed": smi_sample_dict["mol_relaxed"],
            "ab_relaxed": [mol_to_abs(mol) for mol in smi_sample_dict["mol_relaxed"]],
            "mol_unrelaxed": smi_sample_dict["mol"],
            "ab_unrelaxed": smi_sample_dict["ab"],
            "config": config,
        }

    sample_confs_path = (
        f"{config['subfolder']}/{config['sample_subfolder']}_with_relaxed.pkl"
    )
    LOGGER.info("Saving sample_confs_dict to %s", sample_confs_path)
    with open(sample_confs_path, "wb") as f:
        pickle.dump(sample_confs_dict, f)
    LOGGER.info("Saved sample_confs_dict successfully")

    # 3. Prepare true conformer data
    true_confs_dict = {}
    for smi in test_smiles:
        if smi in all_confs_dict:
            true_confs_dict[smi] = {
                "z_stack": [
                    torch.stack(
                        [
                            ab_to_z(ab)
                            for ab in conf.ab_gt_all.view(conf.num_renumberings, -1)
                        ]
                    )
                    for conf in all_confs_dict[smi]
                ],
                "ab": [conf.ab_gt for conf in all_confs_dict[smi]],
                "mol": orig_mol_dict[smi],
            }

    true_confs_dict_path = f"{config['subfolder']}/true_confs_dict_{datetime.now().strftime('%y_%m_%d__%H_%M_%S')}.pkl"
    LOGGER.info("Saving true_confs_dict to %s", true_confs_dict_path)
    with open(true_confs_dict_path, "wb") as f:
        pickle.dump(true_confs_dict, f)
    LOGGER.info("Saved true_confs_dict successfully")
    LOGGER.info("Prepared true conformer data for %d SMILES", len(true_confs_dict))

    try:
        # 4. Create jobs
        jobs: List[Tuple[str, int]] = []
        results: Dict[str, Dict[str, Any]] = {}
        for smi in sample_confs_dict.keys():
            if smi not in true_confs_dict:
                LOGGER.warning(
                    "SMILES %s not present in ground-truth dict; skipping",
                    smi,
                )
                continue
            n_true = len(true_confs_dict[smi]["mol"])
            n_sample = len(sample_confs_dict[smi]["mol_unrelaxed"])
            for i_true in range(n_true):
                jobs.append((smi, i_true))

            results[smi] = {
                "n_true": n_true,
                "n_sample": n_sample,
                "rmsd_unrelaxed": np.nan * np.ones((n_true, n_sample)),
                "rmsd_relaxed": np.nan * np.ones((n_true, n_sample)),
                "rmse_unrelaxed": np.nan * np.ones((n_true, n_sample)),
                "rmse_relaxed": np.nan * np.ones((n_true, n_sample)),
                "ab_diff_unrelaxed": np.nan * np.ones((n_true, n_sample)),
                "ab_diff_relaxed": np.nan * np.ones((n_true, n_sample)),
            }

        # 5. Run jobs in parallel
        num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
        LOGGER.info("Running %s jobs with %s processes", len(jobs), num_processes)
        with Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(true_confs_dict_path, sample_confs_path),
        ) as pool:
            results_list = list(
                tqdm(
                    pool.imap(worker_fn_ab, jobs),
                    total=len(jobs),
                    desc="Calculating metrics",
                )
            )

        # 6. Collect results
        for res in results_list:
            (
                smi,
                i_true,
                rmsds_unrelaxed,
                rmsds_relaxed,
                ab_diffs_unrelaxed,
                ab_diffs_relaxed,
                rmses_unrelaxed,
                rmses_relaxed,
            ) = res
            results[smi]["rmsd_unrelaxed"][i_true, :] = rmsds_unrelaxed
            results[smi]["rmsd_relaxed"][i_true, :] = rmsds_relaxed
            results[smi]["ab_diff_unrelaxed"][i_true, :] = ab_diffs_unrelaxed
            results[smi]["ab_diff_relaxed"][i_true, :] = ab_diffs_relaxed
            results[smi]["rmse_unrelaxed"][i_true, :] = rmses_unrelaxed
            results[smi]["rmse_relaxed"][i_true, :] = rmses_relaxed

        # 7. Calculate and print final stats
        LOGGER.info("%s conformer sets compared", len(results))
        stats = get_stats_and_print("rmsd_unrelaxed", config["thresholds"], results)
        stats.update(get_stats_and_print("rmsd_relaxed", config["thresholds"], results))
        stats.update(
            get_stats_and_print("ab_diff_unrelaxed", config["thresholds"], results)
        )
        stats.update(
            get_stats_and_print("ab_diff_relaxed", config["thresholds"], results)
        )
        stats.update(
            get_stats_and_print("rmse_unrelaxed", config["thresholds"], results)
        )
        stats.update(get_stats_and_print("rmse_relaxed", config["thresholds"], results))
    finally:
        os.remove(true_confs_dict_path)

    return stats


def worker_fn(
    job: Tuple[str, int],
) -> Tuple[str, int, List[float], List[float], List[float], List[float]]:
    """
    Multiprocessing worker for `evaluate`.
    Calculates RMSD and RMSE for one true conformer against all samples.
    """
    smi, i_true = job
    tc = true_confs_dict[smi]["mol"][i_true]

    # RMSD calculation
    rmsds_unrelaxed = [
        AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))  # type: ignore
        for mc in sample_confs_dict[smi]["mol_unrelaxed"]
    ]
    rmsds_relaxed = [
        AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))  # type: ignore
        for mc in sample_confs_dict[smi]["mol_relaxed"]
    ]

    # RMSE calculation
    true_z_stack = true_confs_dict[smi]["z_stack"][i_true]
    rmses_unrelaxed = [
        calc_min_rmse(mol_to_z(mc), true_z_stack)
        for mc in sample_confs_dict[smi]["mol_unrelaxed"]
    ]
    rmses_relaxed = [
        calc_min_rmse(mol_to_z(mc), true_z_stack)
        for mc in sample_confs_dict[smi]["mol_relaxed"]
    ]
    return (smi, i_true, rmsds_unrelaxed, rmsds_relaxed, rmses_unrelaxed, rmses_relaxed)


def evaluate(config):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
        ],
    )

    if config["experiment"] == "rdkit":
        with open(f"{config['log_dir']}.pkl", "rb") as f:
            sample_confs_dict = pickle.load(f)

    elif config["experiment"] == "geodiff":
        sampling_path = f'comparison_algorithms/geodiff_samples/geodiff_samples_random_split_{config["seed"]}.pkl'
        sample_confs_dict = pickle.load(open(sampling_path, "rb"))

    elif config["experiment"] == "ml-mcf":
        sampling_path = f'comparison_algorithms/mcf_samples/ml-mcf_samples_random_split_{config["seed"]}.pkl'
        sample_confs_dict = pickle.load(open(sampling_path, "rb"))
    else:
        raise ValueError(f"Unknown experiment type: {config['experiment']}")

    # if sampling_path:
    #     for smi, smi_sample_dict in sample_confs_dict.items():
    #         sample_confs_dict[smi] = {
    #             "mol_relaxed": smi_sample_dict["mol_relaxed"],
    #             "ab_relaxed": [
    #                 mol_to_abs(mol_relaxed)
    #                 for mol_relaxed in smi_sample_dict["mol_relaxed"]
    #             ],
    #             "mol_unrelaxed": smi_sample_dict["mol_unrelaxed"],
    #             "ab_unrelaxed": [
    #                 mol_to_abs(mol_unrelaxed)
    #                 for mol_unrelaxed in smi_sample_dict["mol_unrelaxed"]
    #             ],
    #             "config": config,
    #         }

    #     with open(sampling_path, "wb") as f:
    #         pickle.dump(sample_confs_dict, f)

    with open(config["all_confs_path"], "rb") as f:
        all_confs_dict = pickle.load(f)
    with open(config["mol_dict_path"], "rb") as f:
        orig_mol_dict = pickle.load(f)

    true_confs_dict = {
        smi: {
            "z_stack": [
                torch.stack(
                    [
                        ab_to_z(ab)
                        for ab in conf.ab_gt_all.view(conf.num_renumberings, -1)
                    ]
                )
                for conf in all_confs_dict[smi]
            ],
            "ab": [conf.ab_gt for conf in all_confs_dict[smi]],
            "mol": orig_mol_dict[smi],
        }
        for smi in sample_confs_dict.keys()
        if smi in all_confs_dict
    }

    with open(config["split_json_path"], "r") as f:
        split_json = json.load(f)
    split = "val" if "val" in config["experiment"] else "test"
    orig_mol_dict = {
        smi: value
        for smi, value in orig_mol_dict.items()
        if (smi in set(split_json[split]))
        and (Chem.MolFromSmiles(smi).GetNumAtoms() != 4)
    }

    logging.info("True SMILES list: %s", sorted(orig_mol_dict.keys()))
    logging.info("Sample SMILES list: %s", sorted(sample_confs_dict.keys()))
    if set(sample_confs_dict.keys()) ^ set(orig_mol_dict.keys()):
        missing = set(orig_mol_dict.keys()) - set(sample_confs_dict.keys())
        extra = set(sample_confs_dict.keys()) - set(orig_mol_dict.keys())
        raise ValueError(
            f"Mismatch between sample and original conformers!\nMissing SMILES: {sorted(missing)}\nExtra SMILES: {sorted(extra)}"
        )
    # Create RMSD jobs
    jobs, results = [], {}
    for smi in sample_confs_dict.keys():
        n_true = len(orig_mol_dict[smi])
        n_sample = len(sample_confs_dict[smi]["mol_unrelaxed"])
        logging.info(f"For {smi}, n_true = {n_true}, n_sample = {n_sample}")

        n_expected = min(config["max_confs_per_smi"], 2 * n_true)
        if n_sample != n_expected:
            if n_sample < n_expected:
                raise ValueError(
                    f"Expected {n_expected} samples for {smi}, but got {n_sample}, I need more!"
                )
            logging.info(
                f"Expected {n_expected} samples for {smi}, but got {n_sample}, cutting off at {n_expected} conformers"
            )
            # Uniformly truncate all sample lists to n_expected
            for key in ["mol_unrelaxed", "ab_unrelaxed", "mol_relaxed", "ab_relaxed"]:
                if key in sample_confs_dict[smi]:
                    sample_confs_dict[smi][key] = sample_confs_dict[smi][key][
                        :n_expected
                    ]
            n_sample = n_expected

        for i_true in range(n_true):
            jobs.append((smi, i_true))

        results[smi] = {
            "n_true": n_true,
            "n_sample": n_sample,
            "rmsd_unrelaxed": np.nan * np.ones((n_true, n_sample)),
            "rmsd_relaxed": np.nan * np.ones((n_true, n_sample)),
            "rmse_unrelaxed": np.nan * np.ones((n_true, n_sample)),
            "rmse_relaxed": np.nan * np.ones((n_true, n_sample)),
        }

    # Save temp files for multiprocessing workers
    import tempfile

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        true_confs_path = f.name
        pickle.dump(true_confs_dict, f)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        sample_confs_path_temp = f.name
        pickle.dump(sample_confs_dict, f)

    # Run RMSD jobs
    num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    with Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(true_confs_path, sample_confs_path_temp),
    ) as pool:
        # Run all jobs in parallel and collect all results
        results_list = list(
            tqdm(
                pool.imap_unordered(worker_fn, jobs),
                total=len(jobs),
                desc="Running jobs",
            )
        )

    # Cleanup temp files
    os.remove(true_confs_path)
    os.remove(sample_confs_path_temp)

    # Now merge afterwards in the main process
    for res in results_list:
        smi, i_true, rmsds_unrelaxed, rmsds_relaxed, rmses_unrelaxed, rmses_relaxed = (
            res
        )
        results[smi]["rmsd_unrelaxed"][i_true] = rmsds_unrelaxed
        results[smi]["rmsd_relaxed"][i_true] = rmsds_relaxed
        results[smi]["rmse_unrelaxed"][i_true] = rmses_unrelaxed
        results[smi]["rmse_relaxed"][i_true] = rmses_relaxed

    # Calculate results
    print(len(results), "conformer sets compared")
    thresholds = [round(x, 3) for x in np.arange(0.05, 0.201, 0.025)]
    stats = get_stats_and_print("rmsd_unrelaxed", thresholds, results)
    stats.update(get_stats_and_print("rmsd_relaxed", thresholds, results))
    stats.update(get_stats_and_print("rmse_unrelaxed", thresholds, results))
    stats.update(get_stats_and_print("rmse_relaxed", thresholds, results))
    return stats
