"""
Generating conformers using multiprocessing.

This script loads a trained model, reads a list of SMILES,
and uses a multiprocessing pool to generate conformers in parallel.
"""

# 1. Standard Library Imports
import json
import logging
import os
import pickle
from functools import partial
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue, Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 2. Third-Party Imports
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles
from tqdm import tqdm

# 3. Local Application Imports
from diffusion.sampling import sample
from utils.dataset import InferenceDataset
from utils.utils import get_model, set_seed

ConfigDict = Dict[str, Any]

# --- Global Setup ---

# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")  # type: ignore
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)


def load_parameters(model: torch.nn.Module, config: ConfigDict) -> None:
    """
    Load model weights from a checkpoint file specified in the config.

    Args:
        model: The model to load weights into (in-place).
        config: The configuration dictionary, specifying which model to load.
    """
    if config["gen_model"] == "last_model":
        checkpoint_path = os.path.join(config["log_dir"], "last_model.pt")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "model"
        ]
    else:
        checkpoint_path = os.path.join(config["log_dir"], f"{config['gen_model']}.pt")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()


def _worker_logging_init(log_queue: Queue, determinism_seed: int) -> None:
    """
    Initialize logging for a worker process.

    This function is called by `multiprocessing.Pool` when a new
    worker is spawned. It attaches a `QueueHandler` to the root
    logger, so all logs from the worker are sent to the main
    process's `QueueListener`.

    Args:
        log_queue: The multiprocessing queue to send logs to.
    """
    
    set_seed(determinism_seed)
    queue_handler = QueueHandler(log_queue)
    root_logger = logging.getLogger()
    # Remove any existing handlers
    root_logger.handlers = [queue_handler]
    root_logger.setLevel(logging.INFO)


def process_single_smiles(
    smi: str,
    n_confs_dict: Dict[str, int],
    config: ConfigDict,
    model_info: Union[torch.nn.Module, str],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[int]]:
    """
    Worker function to process a single SMILES string.

    Generates conformers, saves them as XYZ files, and returns the
    generated data.

    Args:
        smi: The SMILES string to process.
        n_confs_dict: A dictionary mapping SMILES to the number of conformers.
        config: The main configuration dictionary.
        model_info: Either a loaded `torch.nn.Module` or a string ('flat'/'prior').

    Returns:
        A tuple containing:
        - The input SMILES string.
        - A dictionary of generated data (or None on failure).
        - A dictionary of ab_paths (or None on failure).
        - A list of flag codes indicating success/failure of generation.
    """
    n_confs = n_confs_dict.get(smi, 0)
    LOGGER.info("Processing %s with %s conformers to generate.", smi, n_confs)

    if n_confs < 1:
        LOGGER.info("Skipping conformer generation for %s (0 requested).", smi)
        return smi, None, None, []

    try:
        dataset = InferenceDataset(smi, n_confs, config)
        ab_paths, dict_entry, flags = sample(
            dataset,
            model_info,
            steps=config["inference_steps"],
        )

        mols: List[Chem.Mol] = dict_entry["mol"]

        # --- Save each conformer as an XYZ file ---
        xyz_folder_name = (
            f"{config['xyz_subfolder']}_{config['sample_subfolder']}"
            if config.get("sample_subfolder")
            else config["xyz_subfolder"]
        )
        xyz_save_dir = Path(config["subfolder"]) / xyz_folder_name
        xyz_save_dir.mkdir(parents=True, exist_ok=True)

        for i, mol in enumerate(mols):
            xyz_file_path = xyz_save_dir / f"{smi}_conf{i}.xyz"
            try:
                rdmolfiles.MolToXYZFile(mol, str(xyz_file_path))
            except Exception:
                LOGGER.exception("Failed to save %s", xyz_file_path)

        # Move results to CPU before returning
        return (
            smi,
            dict_entry,
            ab_paths,
            [int(x) for x in flags],
        )
    except Exception:
        LOGGER.exception("Error processing %s", smi)
    return smi, None, None, []


def get_confs_per_smiles(
    config: ConfigDict,
    smiles: List[str],
    all_confs: Dict[str, Any],
) -> Dict[str, int]:
    """
    Determine how many conformers to generate for each SMILES.

    Args:
        config: The main configuration dictionary.
        smiles: The list of SMILES strings to check.
        all_confs: A dictionary mapping SMILES to their existing conformers.

    Returns:
        A dictionary mapping SMILES to the number of conformers to generate.
    """
    n_confs = {}
    for smi in smiles:
        try:
            n_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()
        except Exception:
            LOGGER.warning("Failed to parse SMILES: %s", smi, exc_info=True)
            n_confs[smi] = 0
            continue

        existing_confs = all_confs.get(smi, -1)
        if existing_confs == -1:
            LOGGER.warning("No conformers found in all_confs_path for: %s", smi)
            n_confs[smi] = 0
        elif not (5 <= n_atoms <= 8):
            # Original code check
            n_confs[smi] = 0
        else:
            n_to_gen = 2 * len(existing_confs)
            max_confs = config.get("max_confs_per_smi", -1)
            if max_confs > 0:
                n_confs[smi] = min(max_confs, n_to_gen)
            else:
                n_confs[smi] = n_to_gen

    return n_confs


def generate(
    config: ConfigDict,
    model: Optional[torch.nn.Module] = None,
) -> None:
    """
    Main entry point for conformer generation.

    Sets up multiprocessing, loads data, calls workers,
    and aggregates/saves results.

    Args:
        config: The main configuration dictionary.
        model: A pre-loaded model (optional). If None, it will be loaded.
        epoch: The epoch number, used for naming result folders.
    """

    # Get model
    model_info: Union[torch.nn.Module, str]
    model = get_model(config)
    load_parameters(model, config)
    model_info = model

    # Get SMILES
    with open(config["all_confs_path"], "rb") as f:
        all_confs = pickle.load(f)

    mode = "test"
    with open(config["split_json_path"], "r") as f:
        smiles: List[str] = json.load(f)[mode]

    LOGGER.info("Found %s SMILES for %s set.", len(smiles), mode)
    n_confs = get_confs_per_smiles(config, smiles, all_confs)
    total_confs = sum(n_confs.values())
    LOGGER.info("Attempting to generate %s conformers in total.", total_confs)

    # Setup paths
    save_dir = Path(config["subfolder"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare for multiprocessing
    conformer_dict: Dict[str, Any] = {}
    all_flags: List[int] = []
    ab_paths_dict: Dict[str, Any] = {}

    max_smiles = config.get("max_generated_smiles", -1)
    smiles_to_process = smiles[:max_smiles] if max_smiles > 0 else smiles

    if not smiles_to_process:
        LOGGER.info("No SMILES requested for generation; exiting early.")
        return

    # Determine number of processes
    # slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    slurm_cpus = 1
    n_processes = min(max(1, slurm_cpus), len(smiles_to_process))
    LOGGER.info("Using %s processes for multiprocessing.", n_processes)

    # Create worker function with fixed parameters
    worker_func = partial(
        process_single_smiles,
        n_confs_dict=n_confs,
        config=config,
        model_info=model_info,
    )

    # Setup logging listener
    worker_log_path = save_dir / "worker.log"
    log_queue: Queue = Queue()
    file_handler = logging.FileHandler(worker_log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
    )
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    LOGGER.info("Worker log listener started. Logging to %s", worker_log_path)

    # --- 6. Run Multiprocessing Pool ---
    results: List[
        Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[int]]
    ] = []
    try:
        with Pool(
            processes=n_processes,
            initializer=_worker_logging_init,
            initargs=(log_queue, config["determinism_seed"]),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_func, smiles_to_process),
                    total=len(smiles_to_process),
                    desc="Generating conformers",
                    dynamic_ncols=True,
                )
            )
    finally:
        # Stop the listener
        listener.stop()
        file_handler.close()
        # Clean up the queue
        try:
            while not log_queue.empty():
                log_queue.get_nowait()
        except Exception:
            pass
        log_queue.close()
        log_queue.join_thread()

    # --- 7. Collect Results ---
    for smi, dict_entry, ab_paths, flags in results:
        if dict_entry is not None:
            conformer_dict[smi] = dict_entry
            ab_paths_dict[smi] = ab_paths
            all_flags.extend(flags)
    LOGGER.info("Done with conformer generation!")
    LOGGER.info("It was tried to generate %s conformers in total.", len(all_flags))

    values, counts = np.unique(all_flags, return_counts=True)
    LOGGER.info("Errors on entire dataset:")
    for val, count in zip(values, counts):
        LOGGER.info("Error code %s: %s", val, count)

    # --- 8. Save Final Dictionaries ---
    sample_confs_dict_path = save_dir / f"{config['sample_subfolder']}.pkl"
    LOGGER.info("Saving samples at %s", sample_confs_dict_path)
    with open(sample_confs_dict_path, "wb") as f:
        pickle.dump(conformer_dict, f)

    ab_paths_dict_path = save_dir / f"{config['ab_paths_subfolder']}.pkl"
    LOGGER.info("Saving ab_paths at %s", ab_paths_dict_path)
    with open(ab_paths_dict_path, "wb") as f:
        pickle.dump(ab_paths_dict, f)
