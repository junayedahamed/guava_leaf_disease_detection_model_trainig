"""
ShuffleNetV2 Training Script for Guava Leaf Disease Classification
Ultra-lightweight model (~2.3M parameters) optimized for speed and efficiency
Dataset: 5 classes, 2000 images per class (10000 total), 224x224 images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import os

# ============================================================================
# Check Required Packages for TFLite Conversion (Early Check)
# ============================================================================
print("🔍 Checking required packages for TFLite conversion...")

# Check ONNX
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
    print("✅ ONNX: Available")
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX: NOT AVAILABLE")
    print("   Install with: pip install onnx onnxruntime")

# Check TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow: Available")
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow: NOT AVAILABLE")
    print("   Install with: pip install tensorflow")

# Check onnx-tf (for ONNX to TensorFlow conversion)
ONNX_TF_AVAILABLE = False
if ONNX_AVAILABLE:
    try:
        from onnx_tf.backend import prepare
        ONNX_TF_AVAILABLE = True
        print("✅ onnx-tf: Available")
    except ImportError:
        ONNX_TF_AVAILABLE = False
        print("❌ onnx-tf: NOT AVAILABLE")
        print("   Install with: pip install onnx-tf")

# Check if all packages are available
TFLITE_CONVERSION_READY = ONNX_AVAILABLE and TF_AVAILABLE and ONNX_TF_AVAILABLE

if not TFLITE_CONVERSION_READY:
    print("\n" + "="*70)
    print("⚠️  WARNING: TFLite conversion packages are missing!")
    print("="*70)
    print("Required packages for TFLite conversion:")
    missing_packages = []
    if not ONNX_AVAILABLE:
        print("  ❌ pip install onnx onnxruntime")
        missing_packages.append("onnx onnxruntime")
    if not TF_AVAILABLE:
        print("  ❌ pip install tensorflow")
        missing_packages.append("tensorflow")
    if not ONNX_TF_AVAILABLE:
        print("  ❌ pip install onnx-tf")
        missing_packages.append("onnx-tf")
    print("\n⚠️  Training will continue, but TFLite conversion will FAIL at the end.")
    print("⚠️  Install missing packages NOW to avoid re-running training:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("="*70)
    print("⏸️  Waiting 5 seconds... (Press Ctrl+C to stop and install packages)")
    print("="*70)
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Stopped by user. Please install packages and re-run.")
        exit(1)
    print("✅ Continuing with training...\n")
else:
    print("✅ All TFLite conversion packages are available!\n")

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data_dir': './guava_dataset',  # TODO: Update this path to your dataset location  #D:\augmented
    'batch_size': 64,  # Larger batch for fast training
    'num_epochs': 30,  # Adjust if needed
    'learning_rate': 0.001,  # Higher LR for quick convergence
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'num_workers': 4,
    'img_size': 224,
    'num_classes': 5,  # 5 classes for guava dataset
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_dir': './models',  # Customizable: Path where models will be saved
    'warmup_epochs': 3,
    'mixup_alpha': 0.2,  # MixUp for better generalization
}

# Create model save directory if it doesn't exist
os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
print(f"📁 Model save directory: {CONFIG['model_save_dir']}")

print(f"🚀 ShuffleNetV2 Training Configuration - Guava Dataset")
print(f"📱 Device: {CONFIG['device']}")
print(f"🎯 Target: High accuracy in {CONFIG['num_epochs']} epochs")
print(f"⚡️ Model: ShuffleNetV2 x1.0 (~2.3M parameters)")
print(f"📊 Dataset: 5 classes, 2000 images per class (10000 total)")
print("="*70)

# ============================================================================
# Data Augmentation & Transforms (Dataset is already preprocessed to 224x224)
# ============================================================================
# Training transforms - only augmentation, no resizing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

# Validation transforms - no preprocessing, images already 224x224
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================================
# Dataset Loading - Guava Dataset with 5 classes
# ============================================================================
print("📂 Loading dataset...")
import os
from PIL import Image

# Custom dataset for Guava leaf disease classification
class CustomGuavaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, limit_per_class=2000):
        self.transform = transform
        self.samples = []
        # 5 classes for guava dataset
        self.classes = ['healthy', 'leaf_blight', 'leaf_miner', 'leaf_spot', 'nutritional disorder']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load images from each class folder
        for idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"⚠️  Warning: Class folder '{cls_dir}' not found!")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(cls_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Limit per class if specified
            if limit_per_class:
                image_files = image_files[:limit_per_class]
            
            # Add to samples
            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                self.samples.append((img_path, idx))
            
            print(f"  ✅ Loaded {len(image_files)} images from '{cls_name}' class")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Dataset is already preprocessed, just load and apply transform
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"⚠️  Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            img = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, label

full_dataset = CustomGuavaDataset(CONFIG['data_dir'], transform=None, limit_per_class=2000)
class_names = full_dataset.classes
print(f"✅ Classes (5 total): {class_names}")
print(f"✅ Total images loaded: {len(full_dataset)}")

# Split dataset into train and validation (80-20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

print(f"✅ Train size: {train_size} | Val size: {val_size}")

# Create separate datasets with proper transforms
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, label = self.dataset.samples[actual_idx]
        try:
            # Dataset is already preprocessed, just load and apply transform
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"⚠️  Error loading image {img_path}: {e}")
            img = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, label
    
    def __len__(self):
        return len(self.indices)

train_dataset = TransformDataset(full_dataset, train_indices.indices, train_transform)
val_dataset = TransformDataset(full_dataset, val_indices.indices, val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                       shuffle=False, num_workers=0, pin_memory=False)

print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ============================================================================
# Model: ShuffleNetV2 x1.0
# ============================================================================
print("\n🔧 Building ShuffleNetV2 model...")
model = models.shufflenet_v2_x1_0(pretrained=True)  # Use pretrained for better results
# Replace the final fully connected layer for 5 classes
model.fc = nn.Linear(model.fc.in_features, CONFIG['num_classes'])
model = model.to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Total parameters: {total_params:,}")
print(f"✅ Trainable parameters: {trainable_params:,}")

# ============================================================================
# Loss, Optimizer, Scheduler
# ============================================================================
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])

# Cosine annealing for fast convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

# ============================================================================
# MixUp Helper
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# TFLite Conversion Function
# ============================================================================
def convert_to_tflite(model, output_path, img_size=224):
    """
    Convert PyTorch model to TFLite format
    Steps: PyTorch -> ONNX -> TensorFlow -> TFLite
    """
    print("\n" + "="*70)
    print("🔄 Converting model to TFLite format...")
    print("="*70)
    
    # Quick check using the flag set at startup
    if not TFLITE_CONVERSION_READY:
        print("❌ Required packages for TFLite conversion are not available.")
        print("   Missing packages:")
        if not ONNX_AVAILABLE:
            print("     - onnx, onnxruntime (pip install onnx onnxruntime)")
        if not TF_AVAILABLE:
            print("     - tensorflow (pip install tensorflow)")
        if not ONNX_TF_AVAILABLE:
            print("     - onnx-tf (pip install onnx-tf)")
        print("\n   Please install missing packages and re-run the script.")
        return False
    
    try:
        # Step 1: Convert PyTorch to ONNX
        print("📦 Step 1: Converting PyTorch to ONNX...")
        model.eval()
        onnx_path = output_path.replace('.tflite', '.onnx')
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size).to(CONFIG['device'])
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11,
            do_constant_folding=True
        )
        print(f"✅ ONNX model saved: {onnx_path}")
        
        # Step 2: Convert ONNX to TensorFlow
        print("📦 Step 2: Converting ONNX to TensorFlow...")
        try:
            # Import is already checked at startup, but verify again
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_path = output_path.replace('.tflite', '_tf')
            tf_rep.export_graph(tf_path)
            print(f"✅ TensorFlow model saved: {tf_path}")
        except Exception as e:
            print(f"❌ Error converting ONNX to TensorFlow: {e}")
            if os.path.exists(onnx_path):
                print(f"   ONNX model is available at: {onnx_path}")
                print("   You can manually convert it later using onnx-tf")
            return False
        
        # Step 3: Convert TensorFlow to TFLite
        print("📦 Step 3: Converting TensorFlow to TFLite...")
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Set target spec for better compatibility
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
                tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ TFLite model saved: {output_path}")
            print(f"📊 Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
            # Cleanup intermediate TensorFlow model (optional - comment out if you want to keep it)
            try:
                import shutil
                if os.path.exists(tf_path):
                    shutil.rmtree(tf_path)
                    print(f"🧹 Cleaned up intermediate TensorFlow model")
            except:
                pass
            
            return True
        except Exception as e:
            print(f"❌ Error converting TensorFlow to TFLite: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error converting to TFLite: {e}")
        print("   Make sure you have installed: onnx, onnxruntime, tensorflow, onnx-tf")
        return False

# ============================================================================
# Training Function
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
        
        # Apply MixUp
        if use_mixup and CONFIG['mixup_alpha'] > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, CONFIG['mixup_alpha'])
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# Validation Function
# ============================================================================
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# Training Loop
# ============================================================================
print("\n" + "="*70)
print("🚀 Starting Training!")
print("="*70)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'learning_rates': []
}

best_val_acc = 0.0
start_time = time.time()
for epoch in range(CONFIG['num_epochs']):
    epoch_start = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n📊 Epoch [{epoch+1}/{CONFIG['num_epochs']}] LR: {current_lr:.6f}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Scheduler step
    scheduler.step()
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rates'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    
    print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"✅ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"⏱️  Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_name = f"best_shufflenetv2_{val_acc:.2f}.pth"
        checkpoint_path = os.path.join(CONFIG['model_save_dir'], checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG
        }, checkpoint_path)
        print(f"💾 Saved best model: {checkpoint_path}")
    
    # Check if target reached
    if val_acc >= 98.0:
        print(f"\n🎯 TARGET REACHED! Val Acc: {val_acc:.2f}% >= 98%")

total_time = time.time() - start_time
print("\n" + "="*70)
print("✅ Training Complete!")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
print("="*70)

# Save history
timestamp = int(time.time())
history_file = f"shufflenetv2_history_{timestamp}.json"
history_path = os.path.join(CONFIG['model_save_dir'], history_file)
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"💾 History saved: {history_path}")

print(f"\n📊 Final Results:")
print(f"   Train Acc: {history['train_acc'][-1]:.2f}%")
print(f"   Val Acc: {history['val_acc'][-1]:.2f}%")
print(f"   Best Val Acc: {best_val_acc:.2f}%")

# ============================================================================
# Convert Best Model to TFLite
# ============================================================================
# Load the best model for conversion
print("\n" + "="*70)
print("📱 Preparing model for TFLite conversion...")
print("="*70)

# Find the best model checkpoint
best_checkpoint = None
checkpoint_files = [f for f in os.listdir(CONFIG['model_save_dir']) 
                   if f.startswith('best_shufflenetv2_') and f.endswith('.pth')]
if checkpoint_files:
    # Sort by validation accuracy in filename
    checkpoint_files.sort(key=lambda x: float(x.split('_')[2].replace('.pth', '')))
    best_checkpoint = os.path.join(CONFIG['model_save_dir'], checkpoint_files[-1])
    print(f"📂 Loading best checkpoint: {best_checkpoint}")
    
    # Load model state
    checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to TFLite
    tflite_filename = f"shufflenetv2_guava_{best_val_acc:.2f}.tflite"
    tflite_path = os.path.join(CONFIG['model_save_dir'], tflite_filename)
    success = convert_to_tflite(model, tflite_path, CONFIG['img_size'])
    
    if success:
        print(f"\n✅ TFLite conversion completed successfully!")
        print(f"📱 Model saved as: {tflite_path}")
    else:
        print(f"\n⚠️  TFLite conversion failed. PyTorch model is still available.")
        print(f"   Required packages: pip install onnx onnxruntime tensorflow onnx-tf")
else:
    print("⚠️  No checkpoint found for TFLite conversion.")