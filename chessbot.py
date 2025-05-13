import argparse
from models import PolicyNet
import torch
import os
import json


parser = argparse.ArgumentParser(description="Chess PolicyNet CLI")
subparsers = parser.add_subparsers(dest="command")

# init
init_parser = subparsers.add_parser("init")
init_parser.add_argument("--name", required=True)
init_parser.add_argument("--embed-dim", type=int, default=32)

# train
train_parser = subparsers.add_parser("train")
train_parser.add_argument("model")
train_parser.add_argument("--epochs", type=int, default=1000)

# simulate
sim_parser = subparsers.add_parser("simulate")
sim_parser.add_argument("model1")
sim_parser.add_argument("model2")
sim_parser.add_argument("--games", type=int, default=10)

args = parser.parse_args()

# Dispatch commands here
if args.command == "init":
    config = {
        "embedding_dim": args.embed_dim,
        "hidden_dim": args.hidden,
        "depth": args.depth
    }

    model = policy_net.PolicyNet(embedding_dim=config["embedding_dim"],
                      hidden_dim=config["hidden_dim"])

    save_dir = f"models/checkpoints"
    config_dir = f"models/configs"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{save_dir}/{args.name}.pt")
    with open(f"{config_dir}/{args.name}.json", "w") as f:
        json.dump(config, f)

    print(f"✅ Model '{args.name}' initialized and saved.")