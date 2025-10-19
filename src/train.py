# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
import torchvision
from torchvision import models, transforms
from tqdm import tqdm
from src.model import get_resnet50_model
from pathlib import Path
from multiprocessing import freeze_support
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# -----------------------------------------------------------------------------#
# CONSTANTS
# -----------------------------------------------------------------------------#
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
DEFAULT_MODEL_PATH = Path("models") / "cifar-10-resnet50.pth"
DEFAULT_ASSETS_DIR = Path("assets")


# -----------------------------------------------------------------------------#
# UTILS
# -----------------------------------------------------------------------------#
def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(model, dataloader, device):
    """Evaluate model and return loss, accuracy, predictions, and labels."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


# -----------------------------------------------------------------------------#
# TRAINING FUNCTION
# -----------------------------------------------------------------------------#
def train_model(
    epochs: int = 20,
    patience: int = 3,
    fast_mode: bool = False,
    output_dir: str | Path = ".",
):
    """
    Train CIFAR-10 model.
    - full mode: trains full ResNet50 with augmentations
    - fast_mode: lightweight ResNet18 subset for CI/CD
    """
    set_seed(42)
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    assets_dir = output_dir / "assets"
    model_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------#
    # FAST-MODE (for CI/CD)
    # -----------------------------#
    if fast_mode:
        print("Running in FAST MODE for CI pipeline integrity check.")

        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        full_trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        # Use small subset for speed
        train_subset = Subset(full_trainset, random.sample(range(len(full_trainset)), 128))
        test_subset = Subset(testset, range(64))

        # Validation split
        val_size = int(0.2 * len(train_subset))
        train_size = len(train_subset) - val_size
        trainset, valset = random_split(train_subset, [train_size, val_size])

        trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
        valloader = DataLoader(valset, batch_size=16)
        testloader = DataLoader(test_subset, batch_size=16)

        # Lightweight model (no pretrained weights)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)

        epochs = 1
        patience = 1
        device = "cpu"  # CI usually has no GPU

    # -----------------------------#
    # FULL-MODE (for local training)
    # -----------------------------#
    else:
        print(f"Running full training on device: {device}")

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        full_trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_eval)

        # Split train/validation
        val_size = int(0.15 * len(full_trainset))
        train_size = len(full_trainset) - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])

        trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
        valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=8)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

        model = get_resnet50_model(num_classes=10).to(device)

    # -----------------------------#
    # OPTIMIZER & TRAINING
    # -----------------------------#
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_train_loss = running_loss / len(trainloader)
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, valloader, device)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "cifar-10-resnet50.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # -----------------------------#
    # FINAL EVALUATION
    # -----------------------------#
    model.load_state_dict(torch.load(model_dir / "cifar-10-resnet50.pth", map_location=device))
    _, test_acc, preds, labels = evaluate(model, testloader, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

    # Save metrics and confusion matrix
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)

    with open(assets_dir / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CIFAR-10 Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(assets_dir / "test_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Artifacts saved to {model_dir} and {assets_dir}")
    return model, test_acc


# -----------------------------------------------------------------------------#
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------#
def main():
    freeze_support()
    train_model(fast_mode=False)


if __name__ == "__main__":
    main()
