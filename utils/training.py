"""Loss utilities and training helpers for PuckerFlow models."""

from datetime import datetime
from typing import Any, Callable, Dict, cast

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

LossFn = Callable[[Tensor, Batch], Tensor]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def symmetric_loss(pred: Tensor, data: Batch) -> Tensor:
    """Compute a symmetry-aware loss across renumbered ring atom orderings.

    Args:
        pred: Flattened predictions of ``ab`` coordinates across the batch.
        data: Batch object containing per-instance pointer arrays and reference
            ``ab`` coordinates for each ring renumbering.
    """

    ab_ptr = [(data.ptr[i] - i * 3) for i in range(len(data.ptr))]
    total_loss: Tensor = pred.new_zeros(1)
    gt_offset = 0
    n_instances = len(ab_ptr) - 1

    for i in range(n_instances):
        start, end = ab_ptr[i], ab_ptr[i + 1]
        ab_len = end - start
        renumberings = data.num_renumberings[i]

        # predictions for this instance: [ab_len]
        pred_block = pred[start:end]

        # ground truth for this instance: [R, ab_len]
        gt_block = data.ab_gt_all[gt_offset : gt_offset + renumberings * ab_len].view(
            renumberings, ab_len
        )
        gt_offset += renumberings * ab_len

        # squared error for each renumbering → [R]
        min_err = torch.norm(gt_block - pred_block, dim=1).min()

        total_loss = total_loss + min_err

    return total_loss.squeeze(0) / n_instances


def regular_loss(pred: Tensor, data: Batch) -> Tensor:
    """Compute the mean squared norm of prediction errors per instance."""

    ab_ptr = [(data.ptr[i] - i * 3) for i in range(len(data.ptr))]
    diff = pred - data.ab_gt
    loss = torch.stack(
        [
            torch.norm(diff[ab_ptr[i] : ab_ptr[i + 1]]) ** 2
            for i in range(len(ab_ptr) - 1)
        ]
    ).mean()
    return loss


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    config: Dict[str, Any],
) -> float:
    """Run a single optimisation epoch and return the average loss."""

    model.train()
    loss_tot = 0.0
    loss_func: LossFn = cast(LossFn, globals().get(config["loss_function"]))

    with tqdm(
        loader, total=len(loader), desc="Training Epoch", dynamic_ncols=True
    ) as pbar:
        for batch in pbar:
            data = batch.to(device)
            optimizer.zero_grad()

            pred = model(data)
            loss = loss_func(pred, data)

            loss.backward()
            optimizer.step()
            loss_tot += loss.item()

            pbar.set_postfix(
                time=datetime.now().strftime("%H:%M:%S"),
                loss=f"{loss.item():.4f}",
            )

    loss_avg = loss_tot / len(loader)
    return loss_avg


@torch.no_grad()
def val_epoch(
    model: torch.nn.Module, loader: DataLoader, config: Dict[str, Any]
) -> float:
    """Evaluate the model without gradient computation."""

    model.eval()
    loss_tot = 0.0
    loss_func: LossFn = cast(LossFn, globals().get(config["loss_function"]))

    with tqdm(
        loader, total=len(loader), desc="Testing Epoch", dynamic_ncols=True
    ) as pbar:
        for batch in pbar:
            data = batch.to(device)
            pred = model(data)
            loss = loss_func(pred, data)
            loss_tot += loss.item()

            pbar.set_postfix(
                time=datetime.now().strftime("%H:%M:%S"),
                loss=f"{loss.item():.4f}",
            )

    loss_avg = loss_tot / len(loader)
    return loss_avg
