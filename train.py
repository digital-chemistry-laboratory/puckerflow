"""Training entry point for PuckerFlow model.

The script reads experiment configuration, instantiates the model and data
pipelines, and delegates to :func:`train` for iterative optimisation. It also
handles logging, checkpointing, and optional sample generation.
"""

import argparse, logging, math, multiprocessing, os, sys, torch, wandb, yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rdkit import RDLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from utils.dataset import construct_loader
from utils.training import val_epoch, train_epoch
from utils.utils import (
    get_model,
    get_optimizer_and_scheduler,
    save_yaml_file,
    set_seed,
)

LRScheduler = Union[_LRScheduler, ReduceLROnPlateau]
ConfigDict = Dict[str, Any]


def train(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ConfigDict,
) -> None:
    """Run the end-to-end optimisation loop with checkpointing.

    Args:
        model: Neural network being optimised.
        optimizer: Optimiser responsible for parameter updates.
        scheduler: Optional learning-rate scheduler (``ReduceLROnPlateau`` or
            standard PyTorch scheduler).
        train_loader: Dataloader yielding mini-batches for optimisation.
        val_loader: Dataloader yielding batches for validation.
        config: Experiment configuration loaded from YAML.
    """

    best_val_loss: float = math.inf
    best_epoch: int = 0

    logging.info("Starting training...")
    for epoch in range(config["epochs"]):
        train_loss: float = train_epoch(model, train_loader, optimizer, config)
        logging.info("Epoch %s: Training Loss %s", epoch, train_loss)
        wandb.log({"train_loss": train_loss}, step=epoch)

        # Validation is triggered at the configured cadence and after the
        # terminal epoch to ensure at least one evaluation.
        validate = (
            epoch % config["val_freq"] == 0 and config["val_freq"] > 0
        ) or epoch == config["epochs"] - 1
        if validate:
            val_loss: float = val_epoch(model, val_loader, config)
            logging.info("Epoch %s: Validation Loss %s", epoch, val_loss)
            wandb.log({"test_loss": val_loss}, step=epoch)

            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(config["log_dir"], "best_model.pt"),
                )

        # Keep the latest training state for resumable fine-tuning.
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler:
            checkpoint["scheduler"] = scheduler.state_dict()

        # Persist periodic checkpoints for monitoring and recoverability.
        periodic_checkpoint = (
            epoch == 0 or epoch % 10 == 0 or epoch == config["epochs"] - 1
        )
        if periodic_checkpoint:
            torch.save(
                checkpoint,
                os.path.join(config["log_dir"], "last_model.pt"),
            )

        # Archive less frequent full snapshots for long running experiments.
        if epoch % 100 == 0 and epoch != 0:
            torch.save(
                model.state_dict(),
                os.path.join(config["log_dir"], f"model_epoch_{epoch}.pt"),
            )


def main() -> None:
    """Parse arguments, initialise services, and start model training.

    The function expects ``--config`` to point to the model configuration YAML
    relative to this file. It initialises logging, random seeds, experiment
    tracking via Weights & Biases, and then launches the training loop.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.info(
        "CPU count: %s",
        os.environ.get("SLURM_CPUS_PER_TASK", "unknown"),
    )

    set_seed(config["determinism_seed"])
    os.makedirs(config["log_dir"], exist_ok=True)

    wandb.init(project="PuckerFlow", name=config["log_dir"], config=config)

    train_loader, val_loader = construct_loader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = get_model(config).to(device)
    wandb.watch(model)

    optimizer, scheduler = get_optimizer_and_scheduler(config, model)
    save_yaml_file(os.path.join(config["log_dir"], "config.yaml"), config)

    train(model, optimizer, scheduler, train_loader, val_loader, config)


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn")

    torch.multiprocessing.set_sharing_strategy("file_system")
    RDLogger.DisableLog("rdApp.*")  # type: ignore
    main()
