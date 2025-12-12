# Figurative Language Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A comprehensive NLP pipeline for detecting metaphors and irony/sarcasm in text using state-of-the-art transformer models.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure)

</div>

---

## Overview

This project implements a complete pipeline for **figurative language detection**, focusing on identifying **metaphors** and **irony/sarcasm** in natural language text. It leverages modern transformer architectures (BERT, RoBERTa, DeBERTa) and explores cross-domain generalization across diverse text sources.

### Key Highlights

- **Multi-Domain Data Collection**: Automated scrapers for Reddit, IMDb reviews, and news articles
- **Multiple Transformer Models**: Support for BERT, RoBERTa, and DeBERTa architectures
- **Cross-Domain Evaluation**: Comprehensive testing of model generalization across domains
- **Class Imbalance Handling**: Dynamic class weighting for imbalanced datasets

---

## Features

### Data Collection
| Source | Description | Script |
|--------|-------------|--------|
| **Reddit** | Posts and comments from multiple subreddits (politics, movies, opinions, etc.) | `scrape_reddit.py` |
| **IMDb** | Movie reviews from trending films via GraphQL API | `collect_imdb.py` |
| **News** | Opinion articles from major outlets (NYT, Guardian, WaPo, etc.) | `collect_news.py` |
| **VUA Corpus** | Academic metaphor corpus with word-level annotations | `prepare_vua.py` |

### Models Supported
- **RoBERTa** (`roberta-base`) - Robust BERT variant optimized for diverse text
- **BERT** (`bert-base-uncased`) - Classic bidirectional encoder
- **DeBERTa** (`microsoft/deberta-v3-base`) - Enhanced attention mechanism with disentangled matrices

### Classification Tasks
- **Binary**: Literal vs. Figurative
- **Multi-class**: Literal, Metaphor, Irony, Both

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/figurative-language-detection.git
cd figurative-language-detection
```

2. **Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API credentials**

Create a `.env` file in the project root:
```env
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=FigurativeLanguageBot/1.0
```

---

## Usage

### 1. Data Collection

#### Scrape Reddit Data
```bash
python -m src.data_collection.scrape_reddit
```
This scrapes posts and comments from configured subreddits with automatic text truncation for transformer compatibility (512 tokens max).

#### Collect IMDb Reviews
```bash
python -m src.data_collection.collect_imdb -n 150 -r 100 -o data/raw/imdb_reviews.csv
```
| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of movies to scrape | 150 |
| `-r` | Reviews per movie | 100 |
| `-o` | Output file path | `data/imdb_reviews.csv` |

#### Scrape News Articles
```bash
python -m src.data_collection.collect_news
```
Collects opinion/analysis pieces from 11 major news sources with recency filtering (last 30 days).

### 2. Prepare VUA Corpus

If using the VUA Metaphor Corpus:
```bash
python -m src.preprocessing.prepare_vua
```
Expected structure:
```
data/
  vua_archive/
    VUA_MPD/
    VUA18/
    VUA18_pos/
    VUA20/
    ...
```

### 3. Training

#### Train Baseline Models on VUA
```bash
cd src/training
python train_baseline.py
```
This trains both RoBERTa and DeBERTa metaphor detectors with:
- Class weighting for imbalanced data
- Early stopping (patience=3)
- Checkpoint saving based on Macro F1

#### Multi-Domain Fine-tuning
```bash
python src/train_multi_domain.py --model roberta --domain all
```

| Flag | Options | Description |
|------|---------|-------------|
| `--model` | `roberta`, `deberta`, `bert` | Model architecture |
| `--domain` | `reddit`, `imdb`, `news`, `all` | Training domain |
| `--prepare-only` | - | Only prepare data splits |

### 4. Cross-Domain Evaluation

Test trained models across domains:
```bash
python test_imdb_cross_domain.py --test_all
```

Or test a specific model:
```bash
python test_imdb_cross_domain.py --model_name roberta
```

---

## Project Structure

```
figurative-language-detection/
├── data/
│   ├── raw/                    # Raw collected data
│   │   ├── imdb_reviews.csv
│   │   ├── news_articles.csv
│   │   └── reddit_data.csv
│   └── processed/              # Annotated & split data
│       ├── vua_train.csv
│       ├── vua_val.csv
│       ├── vua_test.csv
│       ├── imdb_reviews_annotated.csv
│       ├── reddit_annotations.csv
│       └── news_articles_annotated.csv
│
├── src/
│   ├── data_collection/        # Data scrapers
│   │   ├── scrape_reddit.py
│   │   ├── collect_imdb.py
│   │   ├── collect_news.py
│   │   └── annotate_with_gemini.py
│   │
│   ├── preprocessing/          # Data preparation
│   │   ├── prepare_vua.py
│   │   ├── prepare_reddit_data.py
│   │   └── handle_long_texts.py
│   │
│   ├── models/                 # Model definitions
│   │   └── baseline.py         # BERT, RoBERTa, DeBERTa classifiers
│   │
│   ├── training/               # Training scripts
│   │   └── train_baseline.py   # WeightedTrainer & BaselineTrainer
│   │
│   └── train_multi_domain.py   # Multi-domain training pipeline
│
├── results/                    # Evaluation results
│   ├── cross_domain/           # Cross-domain test results
│   │   ├── imdb_bert/
│   │   ├── imdb_roberta/
│   │   └── imdb_deberta/
│   └── multi_domain_comparison.csv
│
├── models/                     # Saved model checkpoints
│   ├── baseline/
│   ├── imdb_finetuned_roberta/
│   └── ...
│
├── config.py                   # API & project configuration
├── requirements.txt            # Python dependencies
├── test_imdb_cross_domain.py   # Cross-domain evaluation script
└── README.md
```

---

## Technical Details

### Text Length Handling

All scrapers include intelligent text truncation to handle transformer token limits:

```python
class TextLengthHandler:
    def smart_truncate(self, text: str) -> str:
        """Sentence-aware truncation preserving semantic units"""
        # Truncates at sentence boundaries when possible
        # Falls back to token-level truncation if needed
```

### Class Weighting

The training pipeline automatically computes balanced class weights:

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, ...):
        # Dynamic class weights based on training distribution
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | Deep learning framework |
| `transformers` | >=4.30.0 | Hugging Face transformers |
| `datasets` | >=2.12.0 | Dataset handling |
| `pandas` | >=1.5.0 | Data manipulation |
| `scikit-learn` | >=1.2.0 | Metrics & evaluation |
| `praw` | >=7.7.0 | Reddit API wrapper |
| `beautifulsoup4` | >=4.12.0 | HTML parsing |
| `newspaper3k` | - | News article extraction |

---

## Future Work

- [ ] Implement multi-task learning for joint metaphor and irony detection
- [ ] Add support for sentence-level attention visualization
- [ ] Explore domain adaptation techniques (e.g., adversarial training)
- [ ] Create web demo with Gradio/Streamlit
- [ ] Fine-tune on multilingual corpora

---

## References

- **VUA Metaphor Corpus**: Steen, G. J., et al. (2010). *A method for linguistic metaphor identification*
- **MIPVU**: Metaphor Identification Procedure Vrije Universiteit
- **RoBERTa**: Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- **DeBERTa**: He, P., et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
