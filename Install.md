# Installation Guide

## Recommended Platform: Kaggle

This tutorial is designed to run on **Kaggle Notebooks** which provides:
- ✅ Free GPU acceleration (NVIDIA Tesla P100/T4)
- ✅ 16GB GPU RAM
- ✅ Pre-installed common ML libraries
- ✅ 30 hours/week of GPU quota

## Quick Start on Kaggle

### Step 1: Create Kaggle Account
1. Go to [kaggle.com](https://www.kaggle.com/)
2. Sign up for a free account
3. Verify your phone number (required for GPU access)

### Step 2: Upload Notebook
1. Click **"Create"** → **"New Notebook"**
2. Or upload `sklearn-machine-learning.ipynb` directly

### Step 3: Enable GPU
1. Click on **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**
3. Click **Save**

### Step 4: Run the Notebook
Execute cells sequentially starting from the package installation cell.

---

## Package Installation

The first cell in the notebook installs all required packages:

```bash
!pip install -U scikit-learn
!pip install -U pandas
!pip install -U matplotlib
!pip install -U seaborn
!pip install -U datasets
!pip install -U transformers
!pip install -U torch torchvision torchaudio
!pip install -U accelerate
!pip install tensorflow
!pip install tf-keras
```

### Package Versions (Recommended)

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥1.3.0 | ML algorithms |
| pandas | ≥2.0.0 | Data manipulation |
| matplotlib | ≥3.7.0 | Visualization |
| seaborn | ≥0.12.0 | Statistical plots |
| datasets | ≥2.14.0 | Hugging Face datasets |
| transformers | ≥4.35.0 | Pre-trained models |
| torch | ≥2.1.0 | Deep learning |
| accelerate | ≥0.24.0 | Training optimization |

---

## Local Installation (Alternative)

If you prefer to run locally with a GPU:

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- cuDNN 8.6+

### Create Virtual Environment

```bash
# Create environment
python -m venv ml-env

# Activate (Linux/Mac)
source ml-env/bin/activate

# Activate (Windows)
ml-env\Scripts\activate
```

### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Other Packages

```bash
pip install scikit-learn pandas matplotlib seaborn
pip install datasets transformers accelerate
pip install tensorflow tf-keras
pip install jupyter notebook
```

### Launch Jupyter

```bash
jupyter notebook sklearn-machine-learning.ipynb
```

---

## Google Colab (Alternative)

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Runtime → Change runtime type → GPU
4. Run cells sequentially

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in training arguments
per_device_train_batch_size=8  # instead of 16
```

### Transformers Version Conflict
```bash
!pip uninstall -y torch_xla
!pip install --upgrade transformers torch
```

### Slow Download on Kaggle
Enable internet access in notebook settings.

---

## Verify Installation

Run this cell to verify your setup:

```python
import torch
import sklearn
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Transformers: {transformers.__version__}")
```

Expected output (on Kaggle):
```
PyTorch: 2.1.0+cu118
CUDA Available: True
GPU: Tesla T4
Scikit-learn: 1.3.2
Transformers: 4.35.0
```
