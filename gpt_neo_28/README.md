# GPT-Neo 28M Training on TinyStories

Train a 28M parameter GPT-Neo model on the TinyStories dataset with:
- Custom 10K vocabulary (reduced from 50K)
- Deterministic data ordering (fixed seed shuffling)
- Frequent checkpointing (every 1% of training)
- Automatic upload to HuggingFace Hub
- Complete data tracking (saves training examples between checkpoints)

## Model Configuration

- **Parameters**: ~28.2M
- **Vocabulary**: 10,000 tokens (top frequent from TinyStories)
- **Context Length**: 512 tokens
- **Window Size**: 256 tokens (for local attention)
- **Architecture**: GPT-Neo with alternating global/local attention
  - Hidden size: 520
  - Layers: 7
  - Attention heads: 10
  - Intermediate size: 2080

## Setup

### 1. Install Dependencies

```bash
cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28
pip install -r requirements.txt
```

### 2. Authenticate with HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace API token when prompted. You can create a token at: https://huggingface.co/settings/tokens

### 3. Build Vocabulary

First, build the custom 10K vocabulary from TinyStories:

```bash
python build_vocab.py --vocab_size 10000 --output vocab_mapping.json
```

This will:
- Load the entire TinyStories dataset
- Count token frequencies
- Select the top 10,000 most common tokens
- Save a mapping file (`vocab_mapping.json`)

Expected runtime: ~10-20 minutes depending on your system.

### 4. Verify Model Configuration

Optionally verify the parameter count:

```bash
python calculate_params.py
```

## Training

### Basic Training Command

```bash
python train_gptneo.py \
  --config config-28M-gptneo.json \
  --vocab_mapping vocab_mapping.json \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --epochs 1 \
  --seed 3407 \
  --checkpoint_frequency 656 \
  --hf_repo_id YOUR_USERNAME/gpt-neo-28m-tinystories \
  --output_dir ./outputs \
  --use_wandb
```

### Command Line Arguments

- `--config`: Path to model config JSON (default: `config-28M-gptneo.json`)
- `--vocab_mapping`: Path to vocabulary mapping (default: `vocab_mapping.json`)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--epochs`: Number of epochs (default: 1)
- `--seed`: Random seed for reproducibility (default: 3407)
- `--checkpoint_frequency`: Save checkpoint every N updates (default: 656 for ~1%)
- `--log_frequency`: Log validation loss every N updates (default: 100)
- `--hf_repo_id`: HuggingFace repo ID for uploading (e.g., `username/model-name`)
- `--output_dir`: Local output directory (default: `./outputs`)
- `--resume_from`: Path to checkpoint to resume training
- `--use_wandb`: Enable Weights & Biases logging

### Training Details

**Expected Training Time**: ~6-8 hours on 2x V100 GPUs

**Checkpoints**:
- Saved every ~656 updates (1% of full epoch)
- Total of ~100 checkpoints per epoch
- Each checkpoint includes:
  - Model weights
  - Optimizer state
  - Training metadata
  - Data subset (raw text stories used between checkpoints)

**Data Tracking**:
- Each checkpoint saves the exact training examples used since the last checkpoint
- Format: JSON with stories and dataset indices
- Allows for perfect reproducibility and data auditing

**Deterministic Training**:
- Fixed seed (default: 3407)
- Deterministic shuffling with NumPy RandomState
- Same data order every run

## Output Structure

```
outputs/
├── logs/
│   └── train_20231029_143022.log
├── checkpoint_656/
│   ├── checkpoint_656.pt
│   ├── metadata_656.json
│   └── data_checkpoint_656.json
├── checkpoint_1312/
│   ├── checkpoint_1312.pt
│   ├── metadata_1312.json
│   └── data_checkpoint_1312.json
...
└── checkpoint_65625_final/
    ├── checkpoint_65625.pt
    ├── metadata_65625.json
    └── data_checkpoint_65625.json
```

## HuggingFace Hub

Checkpoints are automatically uploaded to HuggingFace Hub if `--hf_repo_id` is specified.

Repository structure on HuggingFace:
```
your-username/gpt-neo-28m-tinystories/
├── checkpoint_656/
├── checkpoint_1312/
...
└── checkpoint_65625_final/
```

## Monitoring

### Weights & Biases

If `--use_wandb` is enabled, training metrics will be logged to W&B:
- Training loss
- Validation loss
- Updates per epoch
- Model configuration

### Logs

Training logs are saved to `outputs/logs/train_TIMESTAMP.log`

## Resuming Training

To resume from a checkpoint:

```bash
python train_gptneo.py \
  --resume_from outputs/checkpoint_656/checkpoint_656.pt \
  --hf_repo_id YOUR_USERNAME/gpt-neo-28m-tinystories \
  ...other args...
```

## Vocabulary Coverage

The 10K vocabulary typically covers ~95-98% of tokens in TinyStories. Out-of-vocabulary tokens are mapped to `[UNK]`.

Special tokens:
- PAD: 0
- UNK: 1
- BOS: 2
- EOS: 3

## Notes

- Training is fully deterministic with fixed seed
- Data order is consistent across runs
- Each checkpoint saves ~21,000 training examples (656 updates × 32 batch size)
- Total training examples: ~2.1M (full TinyStories train set)
- Model uses tied embeddings (input and output)

## Citation

TinyStories dataset:
```
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```



 Start Training

  cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28
  sbatch sbatch_train.sh

  Resume from Checkpoint

  sbatch sbatch_train.sh --resume_from outputs/checkpoint_656/checkpoint_656.pt

  Monitor

  # Check job status
  squeue -u $USER

  # Watch output
  tail -f logs/slurm-JOBID.out