
## Supported Datasets

- Oxford Pets
- Caltech101
- Stanford Cars
- Oxford Flowers
- Food101
- FGVC Aircraft
- SUN397
- DTD (Describable Textures)
- EuroSAT
- UCF101



## Usage

### Training CLIP Models

Use the unified CLIP pipeline script to train models on specific datasets:

```bash
python unified_clip_pipeline.py \
  --dataset oxford_pets \
  --model_type ViT-B/32 \
  --batch_size 64 \
  --epochs 30 \
  --learning_rate 3e-4 \
  --output_dir ./clip_models
```

Train on multiple datasets combined:

```bash
python unified_clip_pipeline.py \
  --dataset all \
  --model_type ViT-B/16 \
  --batch_size 32 \
  --epochs 50 \
  --output_dir ./clip_models_combined
```

### Evaluating CLIP Zero-Shot Performance

Evaluate a trained CLIP model:

```bash
python general_clip_evaluation.py \
  --model_path ./clip_models/oxford_pets/clip_best_model.pt \
  --model_type ViT-B/32 \
  --dataset oxford_pets \
  --data_dir ./data \
  --output_dir ./evaluation_results
```


## Project Structure

### CLIP Training Components

- `unified_clip_pipeline.py`: Main script for running the complete CLIP pipeline
- `general_clip_dataset_pipeline.py`: Dataset loading and processing for CLIP
- `general_clip_evaluation.py`: Zero-shot evaluation tools for CLIP models
- `clip_paper_implementation.py`: Implementation of CLIP architecture
- `clip_tokenizer_and_training.py`: Text tokenization and dataset support


