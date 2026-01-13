"""
Run benchmarking for conformer generation over a list of inference steps.

This script iterates through a specified list of `inference_steps`,
running `generate` and `evaluate_with_ab` multiple times (`n_repeats`)
for each step count.

It aggregates the mean and standard deviation of all metrics and
saves the results to a CSV file.
"""

# 1. Standard Library Imports
import argparse, copy, csv
from pathlib import Path
from typing import Any, Dict, List

# 2. Third-Party Imports
import numpy as np
import wandb, yaml

# 3. Local Application Imports
from utils.evaluate_confs import evaluate_with_ab
from utils.generate_confs import generate
from utils.utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark generation performance across inference steps."
    )
    parser.add_argument(
        "--logdirname",
        type=str,
        required=True,
        help="Path to the model's log directory (containing config.yaml)."
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file for generation and evaluation."
    )
    
    args: argparse.Namespace = parser.parse_args()

    # --- 1. Load Base Config ---
    all_stats: Dict[int, Dict[str, float]] = {}
    config_path = args.config_path
    
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    base_seed = config.get('seed', 0)
    base_det_seed = config.get('determinism_seed', base_seed)

    # --- 2. Run Benchmarking Loop ---
    inference_steps_list = [1, 2, 5, 10, 20, 30, 50]
    for steps in inference_steps_list:
        print(f"\n--- Benchmarking: {steps} inference steps ---")

        # Work on a fresh copy of the config to avoid cross-run mutations
        run_config: Dict[str, Any] = copy.deepcopy(config)
        run_config['inference_steps'] = steps
        run_config['sample_subfolder'] = f'benchmarking_{steps}'
        run_config['ab_paths_subfolder'] = f"ab_paths_{run_config['sample_subfolder']}"
        run_config['seed'] = base_seed
        run_config['determinism_seed'] = base_det_seed

        # Seed everything for reproducibility
        set_seed(run_config['seed'])

        run_name = f"benchmark_steps{steps}"
        wandb_run = wandb.init(
            project="PuckerFlow",
            name=run_name,
            config=run_config,
        )

        try:
            print(f"    Running generation...")
            generate(run_config)

            print(f"    Running evaluation...")
            stats: Dict[str, float] = evaluate_with_ab(run_config)
            stats['inference_steps'] = steps

            print(f"    Done. AMR (relaxed): {stats.get('rmsd_relaxed_mean_amr_recall', 'N/A')}")
            wandb.log(stats)
            all_stats[steps] = stats
        finally:
            wandb_run.finish()

    # --- 3. Aggregate and Save Stats ---
    csv_rows: List[Dict[str, Any]] = []
    for steps, stats in all_stats.items():
        if not stats:
            continue

        metric_keys = {
            key
            for key in stats.keys()
            if key != 'inference_steps'
            and isinstance(stats[key], (int, float, np.floating))
        }

        for metric in sorted(metric_keys):
            mean_val = float(stats[metric])
            csv_rows.append(
                {
                    'inference_steps': steps,
                    'metric': metric,
                    'value': mean_val,
                }
            )

    csv_out_path = Path(f"dont_publish/inference_steps_stats_{base_seed}.csv")
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_out_path.open('w', newline='') as csvfile:
        fieldnames = ['inference_steps', 'metric', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved aggregated stats to {csv_out_path}")

