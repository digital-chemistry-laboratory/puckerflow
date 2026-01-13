from utils.evaluate_confs import evaluate
import wandb, yaml, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(project="PuckerFlow", name=config['experiment'], config=config)
    wandb.log(config, step=0)
    stats = evaluate(config)
    wandb.log(stats, step=0)

if __name__ == "__main__":
    main()