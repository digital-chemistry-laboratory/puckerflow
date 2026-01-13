import pickle, wandb, sys, argparse, logging, json, yaml
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
from pathlib import Path
from utils.coordinate_transforms import *
from utils.evaluate_confs import evaluate
from utils.generate_confs import get_confs_per_smiles
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def get_embedder(method: str):
    """
    Get RDKit embedding parameters based on method choice.

    Args:
        method: 'etkdg', 'etkdg_small_torsions', or 'kdg'

    Returns:
        Dictionary of parameters for AllChem.EmbedMolecule
    """

    if method.lower() == "etkdg":
        return AllChem.ETKDGv2()
    elif method.lower() == "etkdg_small_torsions":
        return AllChem.srETKDGv3()
    elif method.lower() == "kdg":
        return AllChem.KDG()
    else:
        raise ValueError(f"Unknown embedding method: {method}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDKit baseline conformer generation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--method",
        choices=["etkdg", "etkdg_small_torsions", "kdg"],
        default="etkdg",
        type=str,
        help="Embedding method: 'etkdg' (default), 'etkdg_small_torsions', or 'kdg'",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set embedding parameters from CLI args
    config["method"] = args.method

    # Set mode to 'test' if not specified
    if "mode" not in config:
        config["mode"] = "test"

    # Set experiment name for evaluate function
    if "experiment" not in config:
        config["experiment"] = f"rdkit_{args.method}_test"

    setup_logging()

    wandb.init(
        project="PuckerFlow",
        name=f"rdkit_{args.method}",
        config=config,
    )

    # Load all conformers from the single pkl file
    with open(config["all_confs_path"], "rb") as f:
        all_confs = pickle.load(f)

    # Extract SMILES only for the specified set (default: test)
    with open(config["split_json_path"], "r") as f:
        smiles = json.load(f)[config.get("mode", "test")]

    logging.info("Generating conformers for %d SMILES", len(smiles))
    n_confs = get_confs_per_smiles(config, smiles, all_confs)

    # Get embedding parameters
    embed_params = get_embedder(config["method"])
    logging.info(
        f"Using method={config['method']}"
    )

    sample_confs_dict = dict()

    for smi in tqdm(smiles, desc="Generating RDKit conformers"):
        if smi not in all_confs:
            raise ValueError(f"SMILES {smi} not found in all_confs")
        num_confs = min(50, n_confs[smi])
        if num_confs == 0:
            raise ValueError(f"No conformers to generate for {smi}")

        logging.info(f"Generating {num_confs} conformers for {smi}")
        sample_confs_dict[smi] = dict()
        base_mol = Chem.MolFromSmiles(smi)
        if base_mol is None:
            logging.error("Failed to parse SMILES %s", smi)
            continue

        try:
            conf_ids = list(
                AllChem.EmbedMultipleConfs(base_mol, num_confs, embed_params)
            )
        except Exception as e:
            logging.error(
                f"Embedding failed for {smi} with method={config['method']}: {e}; retrying with etkdg"
            )
            try:
                fallback_params = get_embedder("etkdg")
                conf_ids = list(
                    AllChem.EmbedMultipleConfs(base_mol, num_confs, fallback_params)
                )
            except Exception as e2:
                raise ValueError(f"Embedding failed for {smi} again: {e2}")

        if not conf_ids:
            raise ValueError(f"No conformers generated for {smi}")

        for conf_id in conf_ids:
            mol_unrelaxed = Chem.Mol(base_mol)
            mol_unrelaxed.RemoveAllConformers()
            mol_unrelaxed.AddConformer(base_mol.GetConformer(conf_id), assignId=True)

            mol_relaxed = Chem.Mol(mol_unrelaxed)
            try:
                AllChem.MMFFOptimizeMolecule(mol_relaxed, confId=0)
            except Exception as e:
                logging.error("MMFF optimization failed for %s: %s; trying UFF", smi, e)
                try:
                    AllChem.UFFOptimizeMolecule(mol_relaxed, confId=0)
                except Exception as e2:
                    logging.error("UFF optimization failed for %s: %s", smi, e2)
                    continue

            sample_confs_dict[smi].setdefault("mol_unrelaxed", []).append(mol_unrelaxed)
            sample_confs_dict[smi].setdefault("mol_relaxed", []).append(mol_relaxed)

    # Save generated conformers
    save_dir = Path(config["log_dir"]).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = f'{config["log_dir"]}.pkl'

    logging.info(f"Saving {len(sample_confs_dict)} SMILES to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(sample_confs_dict, f)

    logging.info("Starting evaluation...")
    stats = evaluate(config)
    wandb.log(stats)
    logging.info("Evaluation complete.")
