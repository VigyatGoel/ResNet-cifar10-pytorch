# 🧠 ResNet (Residual Network) from Scratch

## 📝 Overview

This project implements the ResNet model architecture from the paper  
**["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)** using **PyTorch**, and trains it from scratch on the **CIFAR-10** dataset.

Built from the ground up with:
- Basic Blocks with residual connections
- Complete ResNet20 architecture with 20 layers
- Training and evaluation pipeline
- Command-line customization options

## 🔧 Requirements

- Python 3.13+
- PyTorch 2.0+
- torchvision
- numpy

## ⚙️ Setup (Using `uv`)

Install `uv` (super fast Python package manager):

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install all dependencies:**

```bash
uv sync
```

**Run the project:**
```bash
uv run src/main.py
```

### Command-Line Arguments

You can customize the training process and model hyperparameters using command-line arguments:

*   `--epochs`: Number of training epochs (default: 30)
*   `--batch-size`: Input batch size (default: 128)
*   `--lr`: Learning rate (default: 0.1)
*   `--momentum`: Momentum for SGD (default: 0.9)
*   `--weight-decay`: Weight decay (default: 1e-4)
*   `--data-dir`: Directory for storing dataset (default: './data')
*   `--save-path`: Path to save the trained model (default: 'resnet20_cifar10.pth')
*   `--num-workers`: Number of data loading workers (default: 2)
*   `--device`: Device to use ('cuda', 'mps', 'cpu') (default: 'cuda')
*   `--use-compile`: Use `torch.compile` for potential speedup (flag, default: False)

**Example:**

```bash
uv run src/main.py --epochs 20 --batch-size 64 --lr 0.001 --device cuda --use-compile
```

## 🏗️ Project Structure

```
resnet/
├── .venv/                 # Python virtual environment
├── data/                  # CIFAR-10 dataset (downloaded automatically)
├── src/
│   ├── basic_block.py     # ResNet basic block implementation
│   ├── main.py            # Training entry point
│   ├── resnet20.py        # ResNet20 model architecture
│   ├── resnet20_notebook.ipynb  # Jupyter notebook version
│   └── train.py           # Training functions
├── .gitattributes         # Git attributes configuration
├── .gitignore             # Git ignore configuration
├── .python-version        # Python version specification
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # This documentation file
└── uv.lock                # Dependencies lock file for uv package manager
```

## 📊 Results

### 📋 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 60 | Total training iterations |
| **Batch Size** | 128 | Number of samples per gradient update |
| **Optimizer** | SGD | Stochastic Gradient Descent |
| **Weight Decay** | 0.0001 | L2 regularization strength |
| **Momentum** | 0.9 | SGD momentum factor |

### 📉 Learning Rate Schedule

| Epoch Range | Learning Rate | 
|:-----------:|:-------------:|
| 1-25 | **0.1** |
| 25-40 | **0.001** |
| 40-60 | **0.0001** |

### 🎯 Model Performance

| Metric | Accuracy | Dataset |
|:------:|:--------:|:-------:|
| **Training** | ~95.4% | CIFAR-10 |
| **Test** | ~90.1% | CIFAR-10 |


### 💪 Performance Comparison

| Model | CIFAR-10 Test Accuracy | Parameters | Training Time |
|:-----:|:----------------------:|:----------:|:-------------:|
| ResNet20 (Ours) | ~90.1% | ~0.27M | ~20 mins |
| ResNet20 (Paper) | ~91.25% | ~0.27M | N/A |
| ResNet56 | 93.03% | ~0.85M | N/A |
| ResNet110 | 93.57% | ~1.7M | N/A |

*Training time on a single Tesla T4 GPU

## 🚀 Future Improvements

- Add support for ResNet56 and ResNet110
- Add TensorBoard support for training visualization

## 📚 Citation

If you use this implementation in your research, please cite both the original paper and this repository:

```
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image 
recognition. In Proceedings of the IEEE conference on computer vision and 
pattern recognition (pp. 770-778).
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.