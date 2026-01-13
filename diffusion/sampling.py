"""
Core sampling functions for the diffusion model.

This module contains the primary `sample` function, which generates
conformer trajectories by integrating the flow matching model.
"""

import logging, torch
from typing import List, Tuple, Union, TypedDict, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from utils.coordinate_transforms import perturb_cp_instance, set_pos

if TYPE_CHECKING:
    from utils.dataset import InferenceDataset


# --- Constants ---

LOGGER = logging.getLogger(__name__)


class SamplingResultsDict(TypedDict):
    """Typed dictionary describing the outputs of :func:`sample`."""

    ab: npt.NDArray[np.float64]
    mol: List[Chem.Mol]


def sample_prior(data: Data, scale: float = 1.0) -> torch.Tensor:
    """Sample a prior puckering coordinate vector.

    Args:
        data: PyG ``Data`` object whose ``bl`` tensor encodes the ring length and
            device. Only the tensor shape/device are used.
        scale: Global scaling factor used to shrink or expand the sampled prior.

    Returns:
        1D tensor containing ``ring_size - 3`` Cremer-Pople coefficients.
    """
    size: int = data.bl.shape[0]
    device: torch.device = data.bl.device

    prior_sample = (torch.rand(size - 3, device=device) - 0.5) * 1.6

    # Apply different scales to different 'ab' dimensions
    scales = torch.tensor([1, 1, 0.7, 0.7, 0.5], device=device)
    prior_sample = prior_sample * scales[: size - 3]

    return prior_sample * scale


@torch.no_grad()
def sample(
    inf_dataset: "InferenceDataset",
    model: Union[torch.nn.Module, str],
    steps: int = 50,
) -> Tuple[npt.NDArray[np.float64], SamplingResultsDict, List[int]]:
    """
    Run the sampling process for all conformers in an inference dataset.

    Args:
        inf_dataset: The dataset containing (n_confs) copies of the molecule.
        model: The trained model
        steps: The number of integration steps.
        device: The device to run on.

    Returns:
        A tuple (ab_paths, results_dict, flags):
        - ab_paths: (n_confs, steps + 1, ab_dim) array of all trajectories.
        - results_dict: {'ab': ab_ends, 'mol': mols}
        - flags: List of success/error codes (0 = success).

    Raises:
        ValueError: If inf_dataset is empty.
    """
    # Preparation
    device = "cpu"
    times: torch.Tensor = torch.linspace(0, 1, steps + 1, device=device)

    # Lists to collect results
    ab_ends: List[npt.NDArray[np.float64]] = []
    ab_paths: List[npt.NDArray[np.float64]] = []
    flags: List[int] = []
    mols: List[Chem.Mol] = []

    ab_dim: int = Chem.MolFromSmiles(inf_dataset.smiles).GetNumAtoms() - 3
    max_tries: int = 100

    if len(inf_dataset) < 1:
        raise ValueError(
            f"You cannot expect me to sample 0 conformers for {inf_dataset.smiles}."
        )

    # Iterate over all samples in the dataset
    for data in inf_dataset:
        data = data.to(device)
        tries: int = 0
        success: bool = False

        # Type for a single trajectory
        ab_path_single: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]

        # Sample a valid starting prior
        while not success and tries < max_tries + 1:
            perturbation = sample_prior(data)
            try:
                data.cp_amp, data.cp_ang, data.ab, data.pos = perturb_cp_instance(
                    data, perturbation, reset=True, clip=False
                )
                success = True
            except ValueError:
                tries += 1

        if not success:
            LOGGER.warning("Could not sample valid prior for %s", data.name)
            # Create a NaN-filled path and flag failure
            ab_path_single = np.full((steps + 1, ab_dim), np.nan)
            ab_paths.append(ab_path_single)
            flags.append(1)  # 1 = prior sampling failed
            mols.append(data.mol)  # Append original mol
            ab_ends.append(np.full((ab_dim,), np.nan))
            continue  # Go to the next conformer in inf_dataset

        data.num_graphs = 1
        ab_path_single = [data.ab.detach().cpu().numpy().copy()]

        # Flow matching integration loop
        integration_success = True
        for time_idx, t in enumerate(times[:-1]):
            data.ptr = torch.tensor([0, data.num_nodes], device=device)
            data.t = torch.tensor([t], device=device)
            dt = times[time_idx + 1] - times[time_idx]

            # Calculate the update vector
            ab_updates = dt * (model(data) - data.ab) / (1 - t + 1e-9)

            try:
                # Apply the update
                data.cp_amp, data.cp_ang, data.ab, data.pos = perturb_cp_instance(
                    data, ab_updates, reset=False, clip=True
                )
            except Exception:
                LOGGER.exception(
                    "Failed at step %.4f (time_idx %s) for smi %s",
                    float(t),
                    time_idx,
                    data.name,
                )
                # Create a NaN path and flag failure
                ab_path_single = np.full((steps + 1, ab_dim), np.nan)
                integration_success = False
                break  # Exit the time loop

            ab_path_single.append(data.ab.detach().cpu().numpy().copy())

        # Collect results for this sample
        if integration_success:
            ab_ends.append(data.ab.detach().cpu().numpy().copy())
            set_pos(data.mol, data.pos)
            mols.append(data.mol)
            flags.append(0)  # 0 = success
            ab_paths.append(np.stack(ab_path_single, axis=0))
        else:
            flags.append(2)  # 2 = integration failed
            ab_paths.append(ab_path_single)  # Append the NaN array
            mols.append(data.mol)  # Append original mol
            ab_ends.append(np.full((ab_dim,), np.nan))

    # Final Aggregation
    values, counts = np.unique(flags, return_counts=True)
    if len(inf_dataset) > 1:
        LOGGER.info("%s errors:", inf_dataset.smiles)
        if len(values) == 0:
            LOGGER.info("Not available")
        else:
            for val, count in zip(values, counts):
                LOGGER.info("Error %s: %s", val, count)

    # Stack all paths and ends
    ab_paths_np = np.stack(ab_paths, axis=0)  # (n_molecules, steps+1, ab_dim)
    ab_ends_np = np.stack(ab_ends, axis=0)  # (n_molecules, ab_dim)

    results_dict: SamplingResultsDict = {"ab": ab_ends_np, "mol": mols}

    return ab_paths_np, results_dict, flags


class PuckeringNoiseTransform(BaseTransform):
    """Apply randomized interpolation between ground-truth and prior puckering."""

    def __call__(self, data: Data) -> Data:
        data.t = t = torch.rand(1, device=data.x.device)
        tries, data.flag = 0, -1
        while True:
            try:
                ab_prior = sample_prior(data)
                mu_ab = data.ab_gt * t + ab_prior * (1 - t)
                ab = mu_ab  # Noise can be added here if desired

                # perturb_cp_instance returns a tuple (cp_amp, cp_ang, ab, pos)
                (data.cp_amp, data.cp_ang, data.ab, data.pos) = perturb_cp_instance(
                    data, ab, reset=True, clip=True
                )
                data.flag = 0
                break  # Success
            except Exception as e:
                tries += 1
                if tries % 1000 == 0:
                    LOGGER.error(
                        "Failed puckering interpolation for %s after %s tries\n"
                        "Exception: %s\n"
                        "Mode: %s\n"
                        "ab_gt: %s\n"
                        "cp_amp: %s\n"
                        "cp_ang: %s\n"
                        "pos: %s\n"
                        "bond lengths: %s\n"
                        "bond angles: %s",
                        data.name,
                        tries,
                        e,
                        data.mode,
                        data.ab_gt,
                        getattr(data, "cp_amp", None),
                        getattr(data, "cp_ang", None),
                        getattr(data, "pos", None),
                        data.bl,
                        data.bang,
                    )
                    # This is a critical failure, exit to avoid infinite loop
                    exit(1)
        return data
