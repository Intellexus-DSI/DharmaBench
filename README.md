# DharmaBench: A Benchmark for Buddhist Texts in Sanskrit and Classical Tibetan

**DharmaBench** is a comprehensive benchmark suite for evaluating large language models (LLMs) on classification and detection tasks in historical Buddhist texts written in Sanskrit and Classical Tibetan. The benchmark includes 13 tasks (6 Sanskrit, 7 Tibetan), with 4 shared across both languages.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DharmaBench
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API keys:**
   - Copy `keys.yaml` and add your API keys for the models you want to use
   - Supported providers: OpenAI, Anthropic, Google, Together AI, Cohere

### First Run

1. **Configure your evaluation:**
   - Edit `config_llm_eval.yaml` to set your desired model, task, and parameters
   - See [Configuration Parameters](#configuration-parameters) below for details

2. **Run evaluation:**
```bash
python run_llm_eval.py
```

3. **For training classification models:**
   - Open `train_classification.ipynb` in Jupyter
   - Follow the notebook cells to fine-tune XLM-RoBERTa models

## ⚙️ Configuration Parameters

The `config_llm_eval.yaml` file contains all evaluation settings:

### General Settings
- **`temperature`**: Controls randomness (0.3 for standard, 0.8 for self-consistency)
- **`results_dir`**: Directory to save results (default: `results`)
- **`logs_dir`**: Directory to save logs (default: `logs`)
- **`data_dir`**: Path to dataset directory (default: `./data`)

### Evaluation Settings
- **`prompt_type`**: Type of prompting (`zero_shot` or `few_shot`)
- **`shots`**: Number of examples for few-shot learning (0 for zero-shot)
- **`sc_runs`**: Number of self-consistency runs (1 = no self-consistency)
- **`debug`**: Enable debug mode with limited samples
- **`num_samples`**: Number of samples to use in debug mode

### Model Settings
- **`model`**: Model to evaluate (see supported models below)
- **`use_rate_limiter`**: Enable rate limiting for API calls
- **`requests_per_second`**: API request rate limit
- **`batched`**: Use batched inference (recommended)

### Task Selection
Choose one task to evaluate:
- **Sanskrit tasks**: `SMDS`, `VPCS`, `MCS`, `QUDS`, `RCMS`, `RCDS`
- **Tibetan tasks**: `AACT`, `VPCT`, `SCCT`, `THCT`, `QUDT`, `RCMT`, `SDT`

### Supported Models

#### Fast Models (Recommended for testing)
- `gemini-2.0-flash`
- `gemini-2.5-flash`
- `gpt-4o-mini`
- `claude-3-haiku`

#### High-Performance Models
- `gemini-2.5-pro`
- `claude-3.7-sonnet`
- `claude-4-sonnet`
- `gpt-4o`

#### Open Source Models
- `qwen-72b`
- `deepseek-r1`

## 📊 About the Research

DharmaBench addresses the critical gap in evaluating LLMs on historical Buddhist texts, which present unique challenges:

- **Linguistic complexity**: Sanskrit and Classical Tibetan have rich morphological systems
- **Cultural context**: Buddhist texts require understanding of philosophical concepts
- **Multilingual evaluation**: Cross-lingual comparison between Sanskrit and Tibetan
- **Domain-specific tasks**: Specialized tasks like metre classification and commentary detection

The benchmark includes both **classification tasks** (predicting categories) and **detection tasks** (identifying spans in text), providing comprehensive evaluation across different NLP capabilities.

### Key Features
- **13 diverse tasks** across two classical languages
- **4 cross-lingual tasks** for comparative evaluation
- **Balanced dataset sizes** with train/test splits where available
- **Standardized evaluation** with consistent metrics and protocols

## 📁 Dataset Overview

DharmaBench contains carefully curated datasets for each task:

- **Sanskrit tasks**: 6 tasks covering simile/metaphor detection, quotation detection, commentary analysis, and text classification
- **Tibetan tasks**: 7 tasks including thematic classification, scriptural categorization, and translation origin detection
- **Multilingual tasks**: 4 tasks available in both languages for cross-lingual evaluation

Each dataset includes:
- Standardized JSON format with `id` fields
- Train/test splits where applicable
- Balanced class distributions
- High-quality annotations by domain experts

For detailed information about each task and dataset structure, see [Data README](data/README.md).

## 🏃‍♂️ Running Evaluations

### Basic Evaluation
```bash
python run_llm_eval.py --config_file config_llm_eval.yaml
```

### Custom Parameters
```bash
python run_llm_eval.py --seed 42 --sc_runs 3 --responses_dir ./previous_results
```

### Command Line Options
- `--config_file`: Path to configuration file
- `--seed`: Random seed for reproducibility
- `--sc_runs`: Number of self-consistency runs
- `--responses_dir`: Directory with existing responses to reprocess

## 📈 Results and Metrics

Results are saved in the `results/` directory with:
- **`metrics.json`**: Overall performance metrics
- **`classification_report.json`**: Detailed classification report
- **`results.tsv`**: Per-sample predictions and ground truth
- **`responses.json`**: Raw model responses
- **`config.yaml`**: Configuration used for the run

## 🔬 Training Classification Models

The `train_classification.ipynb` notebook provides:
- XLM-RoBERTa fine-tuning for classification tasks
- Cross-lingual transfer learning between Sanskrit and Tibetan
- Comprehensive evaluation and visualization
- Model comparison and analysis

## 📜 Citation

If you use DharmaBench in your research, please cite:

```bibtex
@article{dharmabench2024,
  title={DharmaBench: A Benchmark for Buddhist Texts in Sanskrit and Classical Tibetan},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

- **Data**: CC BY 4.0
- **Code**: Apache 2.0

## 📫 Contact

For questions or contributions, please contact: golankai@gmail.com

---

For detailed dataset information, see [Data README](data/README.md).
