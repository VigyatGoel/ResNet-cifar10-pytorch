# ResNet (Residual Network) from Scratch

This project implements the ResNet modelarchitecture from the paper  
**[“Deep Residual Learning for Image Recognition”](https://arxiv.org/abs/1512.03385)** using **PyTorch**, and trains it from scratch on the **CIFAR-10** dataset.

Built from the ground up — Basic Blocks, Residual connections, ResNet20 with 20 layers.

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

*   `--epochs`: Number of training epochs (default: 10)
*   `--batch-size`: Input batch size (default: 32)
*   `--lr`: Learning rate (default: 3e-4)
*   `--weight-decay`: Weight decay (default: 1e-4)
*   `--data-dir`: Directory for storing dataset (default: './data')
*   `--save-path`: Path to save the trained model (default: 'vit_cifar10_state.pth')
*   `--num-workers`: Number of data loading workers (default: 2)
*   `--device`: Device to use ('cuda', 'mps', 'cpu') (default: 'cuda')
*   `--use-compile`: Use `torch.compile` for potential speedup (flag, default: False)

**Example:**

```bash
uv run src/main.py --epochs 20 --batch-size 64 --lr 1e-4 --device cuda --use-compile
```