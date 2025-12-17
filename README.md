# Machine Learning Tutorial

A comprehensive machine learning tutorial using scikit-learn, Hugging Face Transformers, and deep learning models.

## Overview

This notebook-based tutorial covers fundamental to advanced machine learning concepts, from basic classification with scikit-learn to fine-tuning transformer models for NLP tasks.

## Environment

**Platform:** Kaggle Notebooks (for sufficient GPU resources)

Kaggle provides free GPU acceleration which is essential for:
- Training deep learning models
- Fine-tuning transformer models (BERT, GPT-2, ViT)
- Running inference on large pre-trained models

## Topics Covered

### Scikit-Learn Fundamentals
- **Lesson 1-2:** Iris dataset exploration and classification
- **Lesson 3:** Wine dataset and MNIST data loading
- **Lesson 4:** Data preprocessing (StandardScaler, OneHotEncoder)
- **Lesson 5:** Model training, GridSearchCV hyperparameter tuning
- **Breast Cancer Classification:** Complete Random Forest workflow

### Deep Learning & Transformers
- **Lesson 6:** Decision Trees with Hugging Face datasets
- **Lesson 7:** Image Classification with Vision Transformer (ViT)
- **Lesson 8:** Text Generation with GPT-2
- **Lesson 9:** BERT embeddings
- **Lesson 10-12:** Tokenization, Sentiment Analysis, Text Summarization
- **Lesson 13-15:** Fine-tuning BERT for text classification (IMDB dataset)

## Quick Start

1. Upload the notebook to [Kaggle](https://www.kaggle.com/)
2. Enable GPU acceleration in notebook settings
3. Run the cells sequentially

## Project Structure

```
├── machine-learning.ipynb   # Main tutorial notebook
├── README.md                        # This file
├── Architecture.md                  # System architecture documentation
├── Install.md                       # Installation instructions
└── my_model/                        # Fine-tuned model output (generated)
```

## Models Used

| Model | Task | Source |
|-------|------|--------|
| Random Forest | Classification | scikit-learn |
| Decision Tree | Classification | scikit-learn |
| ViT (google/vit-base-patch16-224) | Image Classification | Hugging Face |
| GPT-2 | Text Generation | Hugging Face |
| BERT | Embeddings & Classification | Hugging Face |
| DistilBERT | Sentiment Analysis | Hugging Face |
| BART | Text Summarization | Hugging Face |

## License

This project is for educational purposes.

## Author

Created for learning machine learning with Python.
