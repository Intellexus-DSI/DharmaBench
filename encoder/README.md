# Transformer Model Fine-tuning Tool

A configuration-driven fine-tuning utility for Hugging Face Transformer models supporting classification and extraction tasks.

## Setup

Install dependencies using the provided environment file:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

## Configuration

Create a YAML configuration file with your task parameters.

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Hugging Face model identifier |
| `train_path` | Path to training data (CSV/JSON) |
| `test_path` | Path to test data (CSV/JSON) |
| `text_column` | Name of text field in dataset |
| `label_column` | Name of label field in dataset |
| `task` | Task type: `classification` or `extraction` |
| `task_name` | Identifier for this experiment |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `validation_path` | Separate validation file | Same as `test_path` |
| `text_pair_column` | Second text column for pair tasks | `null` |
| `test_size` | Test split fraction (JSON only) | `0.1` |
| `val_size` | Validation split fraction (JSON only) | `0.1` |
| `seed` | Random seed (int or list) | `42` |
| `train_batch_size` | Training batch size | `16` |
| `eval_batch_size` | Evaluation batch size | `16` |
| `max_length` | Maximum sequence length | `256` |
| `learning_rate` | Learning rate | `5e-5` |
| `weight_decay` | Weight decay | `0.0` |
| `num_train_epochs` | Number of training epochs | `3` |
| `save_strategy` | Save strategy (`steps` or `epoch`) | `epoch` |
| `eval_steps` | Evaluation frequency (steps) | `200` |
| `save_steps` | Checkpoint frequency (steps) | `200` |
| `logging_steps` | Logging frequency (steps) | `1000` |
| `load_best_model_at_end` | Load best checkpoint after training | `false` |
| `metric_for_best_model` | Metric for model selection | `accuracy` |
| `greater_is_better` | Metric direction | `true` |
| `report_to` | Logging integrations | `["wandb"]` |
| `fp16` | Enable mixed precision training | `false` |
| `output_dir` | Base output directory | `./outputs` |

### Example Configuration

```yaml
# bert-base-buddhist-sanskrit-v2.yaml

# MODEL & DATA
model_name_or_path: Matej/bert-base-buddhist-sanskrit-v2
train_path: path/to/train.csv
test_path: path/to/test.csv
text_column: text
label_column: label
test_size: 0.1
val_size: 0.1

# OUTPUT
task_name: your_task_name
task: classification

# RANDOMNESS
seed: [42, 0, 1, 32, 52]

# BATCH & SEQ LENGTH
train_batch_size: 16
eval_batch_size: 16
max_length: 512

# OPTIMIZATION
learning_rate: 5e-5
weight_decay: 0.0
num_train_epochs: 6

# CHECKPOINT & EVAL STRATEGY
save_strategy: epoch
eval_steps: 200
save_steps: 200
logging_steps: 1000

# BEST-MODEL LOADING
load_best_model_at_end: false
metric_for_best_model: f1
greater_is_better: true

# LOGGING & INTEGRATIONS
report_to: []
fp16: false
```

## Running the Script

### Basic Usage

```bash
python main.py --config_file config.yaml
```

### Example Commands

```bash
# Single configuration
python main.py --config_file bert-base-buddhist-sanskrit-v2.yaml

# Different model
python main.py --config_file cino-v2-base.yaml

# Custom path
python main.py --config_file configs/my_config.yaml
```

## Multi-Seed Training

To evaluate across multiple random seeds, specify a list:

```yaml
seed: [42, 0, 1, 32, 52]
```

The script will train for each seed and compute average and standard deviation of metrics.

## Output Structure

```
outputs/
├── {task_name}_{model_id}/
│   ├── config_run.json
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── logs/
│   └── checkpoint-{step}/
├── {task_name}_{model_id}.metric.json
└── all_runs.csv
```
