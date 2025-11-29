import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CancerDataset(Dataset):
    """Custom Dataset for Cancer MRI images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleCNN(nn.Module):
    """Simple 3-layer CNN for binary cancer classification"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Convolutional layers: 32 -> 64 -> 128 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # Input images are 224x224, after 3 pooling layers: 224/8 = 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def load_data(data_dir, val_size=0.1):
    """Load pre-split cancer dataset from train_images and test_images

    Images are labeled by filename:
    - Files starting with 'Cancer' -> class 0 (Cancer)
    - Files starting with 'Not Cancer' -> class 1 (Not Cancer)
    """
    data_path = Path(data_dir)
    train_dir = data_path / 'train_images'
    test_dir = data_path / 'test_images'

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")

    # Binary classification: Cancer (0) vs Not Cancer (1)
    class_names = ['Cancer', 'Not Cancer']

    print(f"Found {len(class_names)} classes: {class_names}")

    # Load training images - filenames contain the label
    train_paths = []
    train_labels = []
    for img_path in train_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']:
            train_paths.append(str(img_path))
            # Label based on filename
            if img_path.name.lower().startswith('not cancer'):
                train_labels.append(1)  # Not Cancer
            else:  # starts with 'Cancer'
                train_labels.append(0)  # Cancer

    print(f"Training images found: {len(train_paths)}")
    print(f"  - Cancer: {train_labels.count(0)}")
    print(f"  - Not Cancer: {train_labels.count(1)}")

    # Load test images
    test_paths = []
    test_labels = []
    for img_path in test_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']:
            test_paths.append(str(img_path))
            # Label based on filename
            if img_path.name.lower().startswith('not cancer'):
                test_labels.append(1)  # Not Cancer
            else:  # starts with 'Cancer'
                test_labels.append(0)  # Cancer

    print(f"Test images found: {len(test_paths)}")
    print(f"  - Cancer: {test_labels.count(0)}")
    print(f"  - Not Cancer: {test_labels.count(1)}")

    # Split training data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def main():
    # Hyperparameters
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224

    # Minimal data transforms - just resize and normalize
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load data
    print("Loading data...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names = load_data(DATA_DIR)

    # Create datasets
    train_dataset = CancerDataset(train_paths, train_labels, train_transform)
    val_dataset = CancerDataset(val_paths, val_labels, test_transform)
    test_dataset = CancerDataset(test_paths, test_labels, test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    # Initialize model, loss, and optimizer
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel architecture:\n{model}\n")

    # Training loop
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model (Val Acc: {val_acc:.2f}%)")

        print()

    # Load best model and evaluate on test set
    print("Evaluating best model on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    model.eval()
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")


if __name__ == '__main__':
    main()
