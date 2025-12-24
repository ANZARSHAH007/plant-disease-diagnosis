"""
Quick Retrain Script - Better Accuracy
Retrain the classifier with 15 epochs for production-level accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Configuration
TRAIN_DIR = 'dataset/crops/plantdoc_train'
TEST_DIR = 'dataset/crops/plantdoc_test'
MODEL_SAVE_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 15  # Full training
LR = 0.001
IMG_SIZE = 224
device = torch.device('cpu')

print("="*70)
print("RETRAINING CLASSIFIER FOR BETTER ACCURACY")
print("="*70)
print(f"Epochs: {EPOCHS}")
print(f"Device: {device}")
print(f"Learning Rate: {LR}")
print("="*70)

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"✓ Classes: {num_classes}")
print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")

# Model
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

print("\nInitializing model...")
classifier = DiseaseClassifier(num_classes=num_classes).to(device)
print(f"✓ Model loaded: {sum(p.numel() for p in classifier.parameters()):,} parameters")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    return running_loss / total, 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / total, 100 * correct / total

# Training loop
print(f"\n{'='*70}")
print(f"STARTING TRAINING - {EPOCHS} EPOCHS")
print(f"{'='*70}\n")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*70}")
    
    train_loss, train_acc = train_epoch(classifier, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(classifier, test_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    print(f"\n📊 Results:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'class_names': class_names,
            'val_acc': val_acc,
            'epoch': epoch + 1
        }, os.path.join(MODEL_SAVE_DIR, 'disease_classifier.pth'))
        print(f"  ✅ Best model saved! Accuracy: {val_acc:.2f}%")

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"💾 Model saved: {MODEL_SAVE_DIR}/disease_classifier.pth")
print(f"{'='*70}\n")

print("🔄 Restart the web application to use the new model!")
