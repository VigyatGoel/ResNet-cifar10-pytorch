import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet20 import ResNet20
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from train import train, evaluate

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def main(args):
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

    device_name = args.device.lower()

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = ResNet20(num_classes=10).to(device)

    if args.use_compile:
        if device.type == "mps":
            print("torch.compile is not supported on MPS — skipping.")
        else:
            model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    print("Starting training...")
    train(model, train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
    print("Starting evaluation...")
    evaluate(model, test_loader, device)

    print(f"Saving model to {args.save_path}")
    torch.save(model, args.save_path)
    print("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-20 on CIFAR-10')

    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')

    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for storing dataset')
    parser.add_argument('--save-path', type=str, default='resnet20_cifar10.pth', help='Path to save the trained model')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of subprocesses for data loading')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to run the model on: "cuda" for Nvidia GPU, "mps" for Apple Silicon GPU, or "cpu" for CPU')
    parser.add_argument('--use-compile', action='store_true', help="Use torch.compile for model acceleration")
    args = parser.parse_args()
    main(args)