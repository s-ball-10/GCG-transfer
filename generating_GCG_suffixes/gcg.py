import os
import json
import torch
import nanogcg
import argparse
import jailbreakbench as jbb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def main(args):

    # Create GCG config and run GCG
    config = nanogcg.GCGConfig(num_steps=args.num_steps, seed=args.seed, early_stop=True)
    result = nanogcg.run(args.model, args.tokenizer, args.goal, args.target, config)

    # Compile results
    results = {
        "index": args.index,
        "seed": args.seed,
        "goal": args.goal,
        "target": args.target,
        "behavior": args.behavior,
        "category": args.category,
        "num_steps": args.num_steps,
        "best_loss": result.best_loss,
        "best_string": result.best_string,
        "losses": result.losses,
        "strings": result.strings,
    }

    # Save results to output directory
    root = f"results/{args.model_name}"
    output_dir = f"{root}/index-{args.index:04d}/seed-{args.seed:04d}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--end_seed", type=int, default=99)
    args = parser.parse_args()
    
    if args.index < 0 or args.index > 99:
        raise ValueError("[ERROR] index must be between 0 and 99")

    if args.end_seed < 0 or args.end_seed > 99:
        raise ValueError("[ERROR] end_seed must be between 0 and 99")

    # Load the HF model and tokenizer
    args.model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load jailbreakbench dataset
    dataset = jbb.read_dataset()
    args.behavior = dataset.behaviors[args.index]
    args.goal = dataset.goals[args.index]
    args.target = dataset.targets[args.index]
    args.category = dataset.categories[args.index]

    # Run GCG for each seed
    for seed in range(0, args.end_seed + 1):
        seed_path = f"results/{args.model_name}/index-{args.index:04d}/seed-{seed:04d}/results.json"
        if not os.path.exists(seed_path):
            print(f"[INFO] Running GCG for seed {seed}, index {args.index}")
            args.seed = seed
            main(args)
