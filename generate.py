"""
Entry point for running conformer generation and evaluation from a trained model.

This script loads a model's training config (`config.yaml`) and a separate
generation config (`generate_config.yaml`), merges them, initializes
wandb, and then runs the generation and/or evaluation steps.
"""

import argparse
import logging

import wandb
import yaml
import sys

from utils.evaluate_confs import evaluate_with_ab
from utils.generate_confs import generate
from utils.utils import set_seed

if __name__ == "__main__":
    """
    Main execution block:
    1. Parses the --log_dir argument.
    2. Loads and merges configuration files.
    3. Sets the random seed for reproducibility.
    4. Initializes a Weights & Biases (wandb) run.
    5. Runs generation (optional) and evaluation.
    6. Logs final statistics to wandb.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    LOGGER = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run conformer generation and evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Directory containing the model logs and config files.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # If config has determinism_seed, set this, otherwise use seed
    if "determinism_seed" in config:
        set_seed(config["determinism_seed"])
    elif "seed" in config:
        set_seed(config["seed"])
    else:
        LOGGER.warning(
            "Warning: No 'determinism_seed' or 'seed' found in config. "
            "Results may not be reproducible."
        )

    # Initialize wandb
    if "flat" in config["log_dir"]:
        wandb_name = f"flat_{config['seed']}"
    elif "prior" in config["log_dir"]:
        wandb_name = f"prior_{config['seed']}"
    else:
        wandb_name = f"rerun_{config['seed']}_cycloflow"

    wandb.init(project="PuckerFlow", name=wandb_name, config=config)

    # Run generation if not skipped
    if not config.get("evaluate_only", False):
        LOGGER.info("Starting conformer generation...")
        generate(config)
    else:
        LOGGER.info("Skipping generation, running evaluation only.")

    # Run evaluation
    LOGGER.info("Starting evaluation...")
    stats = evaluate_with_ab(config)

    # Uncomment for detailed stats printout dictionary
    # LOGGER.info(f"Evaluation stats: {stats}")

    wandb.log(stats)
    LOGGER.info("Run complete.")
