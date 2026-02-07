import os
import gc
import torch
torch.cuda.empty_cache()
import json
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import random

# Import our CLIP implementation
from clip_paper_implementation import create_clip_model
from clip_tokenizer_and_training import SimpleTokenizer
from general_clip_dataset_pipeline import DatasetFactory, CLIPDatasetWrapper, UCF101Dummy


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


class CLIPZeroShotEvaluator:
    def __init__(self, model_path, model_type="ViT-B/32", device=None, batch_size=16):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.batch_size = batch_size
        print(f"Using device: {self.device}, batch size: {self.batch_size}")
        
        # Clear memory before loading model
        clear_gpu_memory()
        
        # Load model
        self.model = create_clip_model(model_type).to(self.device)
        self.model_path = model_path
        
        # Load model weights
        if model_path.endswith(".pt") and not os.path.basename(model_path).startswith("clip_checkpoint"):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Handle checkpoint format
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model from {model_path}")
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # Set up image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def encode_text_prompts(self, class_names, templates, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        # Create prompts for each class using each template
        all_prompts = []
        for template in templates:
            prompts = [template.format(class_name) for class_name in class_names]
            all_prompts.extend(prompts)
        
        # Tokenize all prompts
        text_tokens = torch.zeros(len(all_prompts), 77, dtype=torch.long)
        for i, prompt in enumerate(all_prompts):
            tokens = self.tokenizer.encode(prompt)
            text_tokens[i, :len(tokens)] = torch.tensor(tokens)
        
        # Process in batches
        text_features_list = []
        for i in range(0, len(text_tokens), batch_size):
            batch_tokens = text_tokens[i:i+batch_size].to(self.device)
            
            # Encode prompts
            with torch.no_grad():
                batch_features = self.model.encode_text(batch_tokens)
                text_features_list.append(batch_features.cpu())  # Move back to CPU to save GPU memory
            
            # Optional: clear GPU cache after each batch to prevent memory buildup
            if self.device.type == 'cuda':
                clear_gpu_memory()
        
        # Concatenate results
        text_features = torch.cat(text_features_list, dim=0)
        
        # Reshape to [num_templates, num_classes, embed_dim]
        embed_dim = text_features.shape[-1]
        text_features = text_features.reshape(len(templates), len(class_names), embed_dim)
        
        return text_features.to(self.device)  # Move to device only when needed
    
    def classify_image(self, image, text_features, class_names):
        # Preprocess image if it's a PIL image
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            # For tensor images, ensure they're in the right format
            if image.dim() == 3:  # Single image
                image_tensor = image.unsqueeze(0).to(self.device)
            else:  # Already batched
                image_tensor = image.to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        
        # Calculate similarities
        num_templates = text_features.shape[0]
        similarities = []
        
        for i in range(num_templates):
            # Calculate similarity for each template
            template_text_features = text_features[i]
            similarity = (100.0 * image_features @ template_text_features.T).softmax(dim=-1)
            similarities.append(similarity)
        
        # Average similarities across templates
        similarity = torch.stack(similarities).mean(dim=0)
        
        # Get top predictions
        values, indices = similarity[0].topk(5)
        
        results = []
        for value, index in zip(values, indices):
            results.append({
                "class": class_names[index],
                "confidence": value.item()
            })
        
        return results, similarity[0]
    
    def evaluate_dataset(self, dataset_name, data_dir, class_mapping_path=None, output_dir=None, num_samples=None):
        # Load test dataset
        try:
            test_dataset_raw, dataset_type = DatasetFactory.get_dataset(
                name=dataset_name,
                root=data_dir,
                train=False,
                download=True
            )
            
            # Create CLIP wrapper for the dataset
            test_dataset = CLIPDatasetWrapper(
                dataset=test_dataset_raw,
                transform=self.preprocess,
                dataset_name=dataset_type
            )
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            # For UCF101, use the dummy dataset if real one fails
            if dataset_name.lower() == "ucf101":
                print("Using UCF101Dummy for evaluation")
                dummy_dataset = UCF101Dummy(root=data_dir, train=False, transform=self.preprocess)
                test_dataset = CLIPDatasetWrapper(dummy_dataset, transform=self.preprocess, dataset_name="ucf101")
            else:
                raise
        
        # Load class mapping if provided
        if class_mapping_path and os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
                idx_to_readable = class_mapping['idx_to_readable']
        else:
            # Use the dataset's own class mapping
            idx_to_readable = test_dataset.idx_to_readable
        
        # Special handling for UCF101 dataset
        if dataset_name.lower() == "ucf101":
            # Create a fixed set of class names for UCF101
            class_names = [
                "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
                "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
                # ... additional class names would go here
            ]
            
            # Override the class mapping if needed
            if len(idx_to_readable) < 10:
                print("Using hardcoded UCF101 class names for evaluation")
                # Create a mapping from indices to class names
                idx_to_readable = {str(i): name for i, name in enumerate(class_names)}
                test_dataset.idx_to_readable = {i: name for i, name in enumerate(class_names)}
            
            class_names = list(test_dataset.idx_to_readable.values())
        else:
            # Get sorted class names
            class_names = list(idx_to_readable.values())
        
        # Get templates appropriate for this dataset
        templates = test_dataset.templates
        
        # Clear GPU memory before encoding text
        clear_gpu_memory()
        
        # Encode text prompts
        print("Encoding text prompts...")
        text_features = self.encode_text_prompts(class_names, templates, batch_size=self.batch_size)
        
        # Results tracking
        all_predictions = []
        all_targets = []
        all_probs = []
        top5_correct = 0
        
        # Limit number of samples if specified
        if num_samples and num_samples < len(test_dataset):
            indices = random.sample(range(len(test_dataset)), num_samples)
        else:
            indices = range(len(test_dataset))
        
        # Process test images
        if dataset_name.lower() == "ucf101":
            print(f"Evaluating on {dataset_name} test set using UCF101Dummy")
        else:
            print(f"Evaluating on {dataset_name} test set")
        
        for idx in tqdm(indices):
            try:
                # Get image and class index
                img, _ = test_dataset[idx]
                
                # Get the true class index
                if isinstance(test_dataset.dataset[idx], tuple) and len(test_dataset.dataset[idx]) == 2:
                    _, class_idx = test_dataset.dataset[idx]
                else:
                    # Handle datasets with different return formats
                    sample = test_dataset.dataset[idx]
                    if hasattr(sample, 'items'):  # Dictionary-like object
                        class_idx = sample['label'] if 'label' in sample else sample.get('class', 0)
                    else:
                        class_idx = sample[1] if isinstance(sample, tuple) and len(sample) > 1 else 0
                
                # Convert tensor to int if needed
                if isinstance(class_idx, torch.Tensor):
                    class_idx = int(class_idx.item())
                
                # Classify image
                predictions, probs = self.classify_image(img, text_features, class_names)
                
                # Get top-1 prediction
                pred_class = predictions[0]["class"]
                pred_idx = class_names.index(pred_class)
                
                # Get ground truth class - handle different types of class_idx
                try:
                    if isinstance(class_idx, int):
                        if str(class_idx) in idx_to_readable:
                            true_class = idx_to_readable[str(class_idx)]
                        else:
                            # If not found by string key, try direct lookup
                            true_class = test_dataset.idx_to_readable.get(class_idx, f"class_{class_idx}")
                    elif isinstance(class_idx, str):
                        true_class = idx_to_readable.get(class_idx, f"class_{class_idx}")
                    else:
                        # Fallback for unknown types
                        true_class = f"class_{class_idx}"
                    
                    # If the true class isn't in class_names, handle gracefully
                    if true_class not in class_names:
                        print(f"Warning: True class '{true_class}' not found in class_names. Using index.")
                        true_class = class_names[min(class_idx, len(class_names)-1)]
                    
                    true_idx = class_names.index(true_class)
                except Exception as e:
                    print(f"Error determining true class: {e}")
                    # Use a fallback
                    true_idx = 0
                    true_class = class_names[0]
                
                # Track results
                all_predictions.append(pred_idx)
                all_targets.append(true_idx)
                all_probs.append(probs.cpu().numpy())
                
                # Check if true class is in top-5 predictions
                top5_classes = [p["class"] for p in predictions]
                if true_class in top5_classes:
                    top5_correct += 1
            
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
            
            # Occasionally clear GPU memory to prevent OOM errors
            if idx % 100 == 0 and self.device.type == 'cuda':
                clear_gpu_memory()
        
        # Calculate accuracy
        if len(all_targets) == 0:
            print("No samples were successfully processed. Cannot calculate accuracy.")
            top1_accuracy = 0.0
            top5_accuracy = 0.0
            num_evaluated = 0
        else:
            top1_correct = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
            num_evaluated = len(all_targets)
            top1_accuracy = top1_correct / num_evaluated
            top5_accuracy = top5_correct / num_evaluated
        
        print(f"Zero-shot top-1 accuracy: {top1_accuracy:.4f} ({top1_correct}/{num_evaluated})")
        print(f"Zero-shot top-5 accuracy: {top5_accuracy:.4f} ({top5_correct}/{num_evaluated})")
        
        # Create output directory
        results = {
            'dataset': dataset_name,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'num_samples': num_evaluated,
            'class_names': class_names
        }
        
        if output_dir and num_evaluated > 0:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results to CSV
            results_df = pd.DataFrame({
                "True Class": [class_names[t] for t in all_targets],
                "Predicted Class": [class_names[p] for p in all_predictions],
                "Correct": [p == t for p, t in zip(all_predictions, all_targets)]
            })
            results_df.to_csv(os.path.join(output_dir, f"{dataset_name}_zero_shot_results.csv"), index=False)
            
            # Create confusion matrix
            cm = confusion_matrix(all_targets, all_predictions)
            
            # If there are too many classes, create a reduced confusion matrix
            if len(class_names) > 30:
                # Get most frequent classes
                class_counts = np.bincount(all_targets)
                top_classes = np.argsort(class_counts)[-30:]
                
                # Filter confusion matrix to top classes
                reduced_cm = cm[top_classes, :][:, top_classes]
                reduced_class_names = [class_names[i] for i in top_classes]
                
                plt.figure(figsize=(20, 16))
                sns.heatmap(reduced_cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=reduced_class_names, yticklabels=reduced_class_names)
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.title(f'Confusion Matrix (Top 30 Classes) - Zero-Shot Accuracy: {top1_accuracy:.4f}')
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_reduced_confusion_matrix.png"))
                plt.close()
            
            # Create full confusion matrix if not too large
            if len(class_names) <= 100:
                plt.figure(figsize=(24, 20))
                sns.heatmap(cm, annot=False, cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.title(f'Confusion Matrix - Zero-Shot Accuracy: {top1_accuracy:.4f}')
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"))
                plt.close()
            
            # Classification report
            try:
                report = classification_report(
                    all_targets, 
                    all_predictions, 
                    target_names=class_names, 
                    output_dict=True,
                    zero_division=0  
                )
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(output_dir, f"{dataset_name}_classification_report.csv"))
            except Exception as e:
                print(f"Error generating classification report: {e}")
            
            # Plot per-class accuracy (top 30 classes)
            try:
                plt.figure(figsize=(15, 10))
                per_class_metrics = {}
                for i, class_name in enumerate(class_names):
                    # Get indices where this is the true class
                    class_indices = [j for j, t in enumerate(all_targets) if t == i]
                    if class_indices:
                        # Calculate accuracy for this class
                        class_correct = sum(1 for j in class_indices if all_predictions[j] == i)
                        class_accuracy = class_correct / len(class_indices)
                        per_class_metrics[class_name] = class_accuracy
                
                # Sort and get top 30
                sorted_metrics = sorted(per_class_metrics.items(), key=lambda x: x[1], reverse=True)
                top_classes = sorted_metrics[:30]
                
                plt.bar([x[0] for x in top_classes], [x[1] for x in top_classes])
                plt.xlabel('Class')
                plt.ylabel('Accuracy')
                plt.title(f'Per-Class Accuracy (Top 30) - {dataset_name}')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_per_class_accuracy.png"))
                plt.close()
            except Exception as e:
                print(f"Error generating per-class accuracy plot: {e}")
            
            # Save overall results
            with open(os.path.join(output_dir, f"{dataset_name}_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP zero-shot classification on various datasets")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained CLIP model")
    parser.add_argument("--model_type", type=str, default="ViT-B/32", 
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-S/32", "toy"],
                        help="CLIP model variant used")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["oxford_pets", "caltech101", "stanford_cars", "flowers102", 
                                 "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101", "all"],
                        help="Dataset to evaluate on ('all' to evaluate on all datasets)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where dataset is stored")
    parser.add_argument("--class_mapping", type=str, default=None, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda, cpu, or cuda:n)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for text encoding")
    
    args = parser.parse_args()
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Set device explicitly if provided
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Create evaluator
    evaluator = CLIPZeroShotEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        batch_size=args.batch_size
    )
    
    # Determine datasets to evaluate
    if args.dataset.lower() == "all":
        datasets = ["oxford_pets", "caltech101", "stanford_cars", "flowers102", 
                    "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101"]
    else:
        datasets = [args.dataset]
    
    # Evaluate each dataset
    all_results = {}
    for dataset in datasets:
        print(f"\n=== Evaluating on {dataset} ===\n")
        
        # Try to find dataset-specific class mapping
        class_mapping = args.class_mapping
        if not class_mapping:
            default_mapping = os.path.join(os.path.dirname(args.model_path), "class_mapping.json")
            if os.path.exists(default_mapping):
                class_mapping = default_mapping
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        try:
            # Evaluate
            results = evaluator.evaluate_dataset(
                dataset_name=dataset,
                data_dir=args.data_dir,
                class_mapping_path=class_mapping,
                output_dir=dataset_output_dir,
                num_samples=args.num_samples
            )
            
            all_results[dataset] = results
            
            # Clear memory after each dataset evaluation
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error evaluating dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # If evaluating on multiple datasets, create a summary
    if len(datasets) > 1 and len(all_results) > 0:
        # Create summary dataframe
        summary_data = []
        for dataset, result in all_results.items():
            if isinstance(result, dict) and 'top1_accuracy' in result:
                summary_data.append({
                    "Dataset": dataset,
                    "Top-1 Accuracy": result["top1_accuracy"],
                    "Top-5 Accuracy": result["top5_accuracy"],
                    "Num Samples": result["num_samples"],
                    "Num Classes": len(result["class_names"])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
            
            # Create bar chart of accuracies
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(summary_data))
            width = 0.35
            
            datasets = [d["Dataset"] for d in summary_data]
            top1_accuracies = [d["Top-1 Accuracy"] for d in summary_data]
            top5_accuracies = [d["Top-5 Accuracy"] for d in summary_data]
            
            plt.bar(x - width/2, top1_accuracies, width, label='Top-1 Accuracy')
            plt.bar(x + width/2, top5_accuracies, width, label='Top-5 Accuracy')
            
            plt.xlabel('Dataset')
            plt.ylabel('Accuracy')
            plt.title('CLIP Zero-Shot Accuracy Across Datasets')
            plt.xticks(x, datasets, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(args.output_dir, "multi_dataset_performance.png"))
            plt.close()
            
            # Print summary
            print("\n=== Summary of Results ===")
            print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()