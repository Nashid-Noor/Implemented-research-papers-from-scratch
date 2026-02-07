#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import torch
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_command(command):
    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        return False
    return True


def check_gpu():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU(s):")
        for i in range(device_count):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("No GPU found, will use CPU (training will be slow)")
        return False


def setup_environment(args):
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Log arguments
    log_file = os.path.join(args.output_dir, "pipeline_config.json")
    with open(log_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create a run ID based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If training on all datasets combined, use a special identifier
    if args.dataset.lower() == "all":
        run_id = f"combined_datasets_{args.model_type.replace('/', '_')}_{timestamp}"
    else:
        run_id = f"{args.dataset}_{args.model_type.replace('/', '_')}_{timestamp}"
    
    return run_id


def train_clip(args, run_id):
    
    # Special message for combined datasets
    if args.dataset.lower() == "all":
        print(f"\n=== Training CLIP Model on Combined Datasets ===\n")
    else:
        print(f"\n=== Training CLIP Model on {args.dataset} ===\n")
    
    # Set up output directory for this specific run
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = [
        sys.executable, "general_clip_dataset_pipeline.py",
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--output_dir", output_dir,
        "--val_split", str(args.val_split),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--label_smoothing", str(args.label_smoothing),
        "--restart_epochs", str(args.restart_epochs)
    ]
    
    if args.val_from_train:
        command.append("--val_from_train")
    
    # Run training
    success = run_command(command)
    
    if not success:
        print("Training failed. Check the logs for details.")
        return None
    
    # Return paths to saved models
    best_model_path = os.path.join(output_dir, "clip_best_model.pt")
    final_model_path = os.path.join(output_dir, "clip_final_model.pt")
    class_mapping_path = os.path.join(output_dir, "class_mapping.json")
    
    model_paths = {
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "class_mapping_path": class_mapping_path,
        "output_dir": output_dir,
        "run_id": run_id
    }
    
    # Save model paths for future reference
    with open(os.path.join(args.output_dir, f"{run_id}_model_paths.json"), "w") as f:
        json.dump(model_paths, f, indent=2)
    
    return model_paths


def evaluate_clip(args, model_paths, datasets_to_evaluate=None):
    if model_paths is None:
        print("No model paths provided. Skipping evaluation.")
        return None
    
    print("\n=== Evaluating CLIP Model (Zero-Shot) ===\n")
    
    # Choose which model to evaluate
    model_path = model_paths["best_model_path"] if args.use_best_model else model_paths["final_model_path"]
    
    # When we've trained on combined datasets, we want to evaluate on each individual dataset
    if args.dataset.lower() == "all" and not datasets_to_evaluate:
        datasets_to_evaluate = "all"
    # If specific datasets are not provided, use either the training dataset or 'all'
    elif datasets_to_evaluate is None:
        if args.eval_on_all:
            datasets_to_evaluate = "all"
        else:
            datasets_to_evaluate = args.dataset
    
    # Create eval directory
    eval_dir = os.path.join(args.eval_dir, model_paths["run_id"])
    os.makedirs(eval_dir, exist_ok=True)
    
    # Build command
    command = [
        sys.executable, "general_clip_evaluation.py",
        "--model_path", model_path,
        "--model_type", args.model_type,
        "--dataset", datasets_to_evaluate,
        "--data_dir", args.data_dir,
        "--class_mapping", model_paths["class_mapping_path"],
        "--output_dir", eval_dir,
        "--batch_size", str(args.eval_batch_size)
    ]
    
    if args.eval_samples:
        command.extend(["--num_samples", str(args.eval_samples)])
    
    if args.eval_device:
        command.extend(["--device", args.eval_device])
    
    # Run evaluation
    success = run_command(command)
    
    if not success:
        print("Evaluation failed. Check the logs for details.")
        return None
    
    return eval_dir


def run_multi_dataset_training(args):
    all_datasets = ["oxford_pets", "caltech101", "stanford_cars", "flowers102", 
                   "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101"]
    
    # Ensure args.dataset is "all" for combined training
    args.dataset = "all"
    
    # Set up environment and get run ID
    run_id = setup_environment(args)
    
    # Train on combined datasets
    if not args.skip_train:
        model_paths = train_clip(args, run_id)
    else:
        # Try to find existing combined model
        combined_model_dir = os.path.join(args.output_dir, "combined_datasets")
        if os.path.exists(combined_model_dir):
            # Find most recent model
            model_files = list(Path(combined_model_dir).rglob("clip_best_model.pt"))
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                model_paths = {
                    "best_model_path": str(latest_model),
                    "final_model_path": str(latest_model.parent / "clip_final_model.pt"),
                    "class_mapping_path": str(latest_model.parent / "class_mapping.json"),
                    "output_dir": str(latest_model.parent),
                    "run_id": run_id
                }
                print(f"Found existing combined model at {latest_model}")
            else:
                print("No existing combined model found. Skipping.")
                return
        else:
            print("No combined model directory found. Skipping.")
            return
    
    # Evaluate on all datasets separately
    if not args.skip_eval and model_paths:
        eval_dir = evaluate_clip(args, model_paths, "all")
        
        # Collect results for each dataset
        results = {}
        if eval_dir:
            for dataset in all_datasets:
                # Look for results file
                results_file = os.path.join(eval_dir, dataset, f"{dataset}_results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        dataset_results = json.load(f)
                        results[dataset] = dataset_results
        
        # Create summary of all results
        if results and len(results) > 0:
            create_multi_dataset_summary(results, args.output_dir)


def create_multi_dataset_summary(results, output_dir):
    summary_dir = os.path.join(output_dir, "multi_dataset_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    for dataset, result in results.items():
        if isinstance(result, dict) and "top1_accuracy" in result:
            summary_data.append({
                "Dataset": dataset,
                "Top-1 Accuracy": result["top1_accuracy"],
                "Top-5 Accuracy": result.get("top5_accuracy", 0),
                "Num Samples": result.get("num_samples", 0),
                "Num Classes": len(result.get("class_names", []))
            })
    
    if not summary_data:
        print("No valid results found for summary.")
        return
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(summary_dir, "all_datasets_summary.csv"), index=False)
    
    # Create bar chart of accuracies
    plt.figure(figsize=(14, 8))
    
    datasets = summary_df["Dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35
    
    top1_accuracies = summary_df["Top-1 Accuracy"].tolist()
    top5_accuracies = summary_df["Top-5 Accuracy"].tolist()
    
    plt.bar(x - width/2, top1_accuracies, width, label='Top-1 Accuracy')
    plt.bar(x + width/2, top5_accuracies, width, label='Top-5 Accuracy')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('CLIP Zero-Shot Accuracy on Individual Datasets (Combined Training)')
    plt.xticks(x, datasets, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(summary_dir, "all_datasets_performance.png"))
    plt.close()
    
    # Print summary
    print("\n=== Summary of Results ===")
    print(summary_df.to_string(index=False))
    
    # Save results as JSON
    with open(os.path.join(summary_dir, "all_datasets_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Complete CLIP pipeline for various datasets")
    
    # General parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store dataset")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save models")
    parser.add_argument("--eval_dir", type=str, default="./evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--model_type", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-S/32", "toy"],
                        help="CLIP model variant")
    parser.add_argument("--use_best_model", action="store_true",
                        help="Use best model instead of final model for evaluation")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="oxford_pets",
                        choices=["oxford_pets", "caltech101", "stanford_cars", "flowers102", 
                                "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101", "all"],
                        help="Dataset to train on ('all' for combined training on all datasets)")
    parser.add_argument("--val_from_train", action="store_true",
                        help="Use a portion of training data for validation")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of training data to use for validation (default 0.1)")

    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.2,
                        help="Weight decay for optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.2,
                        help="Fraction of training for warmup")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor for loss function")
    parser.add_argument("--restart_epochs", type=int, default=5,
                        help="Number of epochs between learning rate restarts")
    
    # Evaluation parameters
    parser.add_argument("--eval_on_all", action="store_true",
                        help="Evaluate on all datasets, not just the one used for training")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Number of samples to use for evaluation (None for all)")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_device", type=str, default=None,
                        help="Device to use for evaluation (cpu or cuda)")
    
    # Pipeline control
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training phase")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation phase")
    parser.add_argument("--existing_model", type=str, default=None,
                        help="Path to existing model (used if skip_train is True)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    if not has_gpu and not args.skip_train:
        print("WARNING: Training on CPU will be very slow. Consider using --skip_train.")
    
    # Handle 'all' dataset option for combined training
    if args.dataset.lower() == "all":
        # Run combined training on all datasets
        run_multi_dataset_training(args)
        return
    
    # Setup for single dataset
    run_id = setup_environment(args)
    
    # Initialize model paths
    model_paths = None
    
    # Training phase
    if not args.skip_train:
        model_paths = train_clip(args, run_id)
    elif args.existing_model:
        # Use existing model if provided
        model_dir = os.path.dirname(args.existing_model)
        model_paths = {
            "best_model_path": args.existing_model,
            "final_model_path": args.existing_model,
            "class_mapping_path": os.path.join(model_dir, "class_mapping.json"),
            "output_dir": model_dir,
            "run_id": "existing_" + run_id
        }
    else:
        # Try to find most recent model for this dataset
        dataset_dir = os.path.join(args.output_dir, args.dataset)
        if os.path.exists(dataset_dir):
            # Look for model files
            model_files = list(Path(dataset_dir).rglob("clip_best_model.pt"))
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                model_paths = {
                    "best_model_path": str(latest_model),
                    "final_model_path": str(latest_model.parent / "clip_final_model.pt"),
                    "class_mapping_path": str(latest_model.parent / "class_mapping.json"),
                    "output_dir": str(latest_model.parent),
                    "run_id": "existing_" + run_id
                }
                print(f"Found existing model at {latest_model}")
    
    # Evaluation phase
    if not args.skip_eval and model_paths:
        eval_dir = evaluate_clip(args, model_paths)
        if eval_dir:
            print(f"Evaluation results saved to {eval_dir}")
    
    print("\n=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()