# Architecture Overview

## System Architecture

This tutorial is designed to run on **Kaggle Notebooks** with GPU acceleration for optimal performance.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kaggle Environment                        │
│                         (GPU Enabled)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Jupyter Notebook                        │   │
│  │              sklearn-machine-learning.ipynb               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│  ┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐      │
│  │   Scikit-Learn   │ │  Hugging    │ │     PyTorch     │      │
│  │                  │ │    Face     │ │                 │      │
│  │  - Classifiers   │ │             │ │  - Tensors      │      │
│  │  - Preprocessing │ │ - Datasets  │ │  - GPU Compute  │      │
│  │  - Metrics       │ │ - Models    │ │  - Autograd     │      │
│  │  - Model Select  │ │ - Pipelines │ │                 │      │
│  └──────────────────┘ └─────────────┘ └─────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Layer

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Scikit-Learn   │  Hugging Face   │     External APIs       │
│  Built-in       │  Datasets       │                         │
├─────────────────┼─────────────────┼─────────────────────────┤
│  - load_iris    │  - IMDB         │  - COCO Dataset         │
│  - load_wine    │  - Iris (HF)    │    (images)             │
│  - load_breast_ │                 │                         │
│    cancer       │                 │                         │
│  - fetch_openml │                 │                         │
│    (MNIST)      │                 │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 2. Model Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     Machine Learning Models                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional ML (CPU)          Deep Learning (GPU)          │
│  ┌────────────────────┐        ┌────────────────────┐       │
│  │  RandomForest      │        │  Vision Transformer│       │
│  │  DecisionTree      │        │  (ViT)             │       │
│  │  GridSearchCV      │        │                    │       │
│  └────────────────────┘        │  BERT / DistilBERT │       │
│                                │  GPT-2             │       │
│                                │  BART              │       │
│                                └────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Processing Pipeline

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Data   │───▶│ Preprocessing│───▶│   Training  │───▶│Evaluation│
│  Loading │    │              │    │             │    │          │
└──────────┘    └──────────────┘    └─────────────┘    └──────────┘
                      │                    │                 │
                      ▼                    ▼                 ▼
               ┌─────────────┐     ┌─────────────┐    ┌───────────┐
               │StandardScaler│    │ model.fit() │    │ accuracy  │
               │OneHotEncoder│     │ Trainer()   │    │ confusion │
               │ Tokenizer   │     │             │    │  matrix   │
               └─────────────┘     └─────────────┘    └───────────┘
```

## GPU Utilization

### Why Kaggle GPU?

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| BERT Fine-tuning (IMDB) | ~4 hours | ~20 min | 12x |
| ViT Inference | ~5 sec | ~0.5 sec | 10x |
| GPT-2 Generation | ~2 sec | ~0.2 sec | 10x |

### GPU Memory Usage

```
┌─────────────────────────────────────────────────────────────┐
│                    Kaggle GPU (16GB VRAM)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Loading:     ~2-4 GB                                 │
│  Training Batch:    ~4-8 GB (batch_size=16)                 │
│  Gradient Storage:  ~2-4 GB                                 │
│  ─────────────────────────────                              │
│  Total Peak:        ~8-16 GB                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Classification Pipeline (Scikit-Learn)

```
Raw Data → Train/Test Split → Model Training → Predictions → Metrics
    │                              │                            │
    ▼                              ▼                            ▼
 DataFrame                  RandomForest              accuracy_score
                           .fit(X_train)              confusion_matrix
```

### NLP Pipeline (Hugging Face)

```
Text → Tokenization → Model Inference → Post-processing → Output
  │         │               │                  │             │
  ▼         ▼               ▼                  ▼             ▼
"Hello"  [101,7592]    logits tensor      softmax      "Positive"
```

### Fine-tuning Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Pre-trained │     │   Training   │     │  Fine-tuned  │
│    Model     │────▶│   Process    │────▶│    Model     │
│   (BERT)     │     │  (Trainer)   │     │ (my_model/)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    IMDB      │
                     │   Dataset    │
                     └──────────────┘
```

## Output Artifacts

```
./results/           # Training checkpoints
./my_model/          # Final fine-tuned model
  ├── config.json
  ├── model.safetensors
  ├── tokenizer.json
  └── vocab.txt
```
