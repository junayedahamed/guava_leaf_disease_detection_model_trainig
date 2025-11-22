"""
SqueezeNet training (PyTorch) + ONNX -> SavedModel -> TFLite export pipeline.

Dataset: D:\augmented  (ImageFolder: classes must be subfolders)
Classes: healthy, leaf_blight, leaf_miner, leaf_spot, nutritional disorder
Image size: 224x224
Epochs: 30 (as requested)

Note: This script will save:
 - checkpoints_guava/best_model.pth   (PyTorch weights)
 - onnx_model.onnx                     (ONNX)
 - saved_model/                        (TensorFlow SavedModel)
 - model.tflite                        (TFLite)

If ONNX->SavedModel conversion fails, check onnx/onnx-tf compatibility and TensorFlow install.
"""

import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, classification_report

# -------------------------
# USER CONFIG
# -------------------------
DATASET_PATH = r"D:\augmented"
NUM_CLASSES = 5
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 224

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Model save directory - all models will be saved here
MODEL_SAVE_DIR = r"D:\model_trainig\squeezenet\trained_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # make folder if it doesn't exist

# Define all model save paths
CHECKPOINT_DIR = MODEL_SAVE_DIR  # Use same directory for all saves
best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
onnx_path = os.path.join(MODEL_SAVE_DIR, "onnx_model.onnx")
saved_model_dir = os.path.join(MODEL_SAVE_DIR, "saved_model")
tflite_path = os.path.join(MODEL_SAVE_DIR, "squeezenet_model.tflite")

print(f"📁 Model save directory: {MODEL_SAVE_DIR}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------
# Seed
# -------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------
# Main execution
# -------------------------
if __name__ == '__main__':
    # -------------------------
    # Transforms
    # -------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -------------------------
    # Load dataset (ImageFolder) - Load once
    # -------------------------
    print("Loading dataset from:", DATASET_PATH)
    full_train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_tf)
    full_val_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_tf)
    class_names = full_train_dataset.classes
    print("Detected classes:", class_names)
    assert len(class_names) == NUM_CLASSES, f"Detected {len(class_names)} classes, but NUM_CLASSES={NUM_CLASSES}"

    total_len = len(full_train_dataset)
    train_len = int(TRAIN_RATIO * total_len)
    val_len = int(VAL_RATIO * total_len)
    test_len = total_len - train_len - val_len
    print(f"Total images: {total_len}. Split -> train: {train_len}, val: {val_len}, test: {test_len}")

    indices = list(range(total_len))
    random.shuffle(indices)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)
    test_dataset = Subset(full_val_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------
    # Model: SqueezeNet 1.1 (pretrained)
    # -------------------------
    try:
        from torchvision.models import SqueezeNet1_1_Weights
        model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    except (TypeError, AttributeError):
        # Fallback for older torchvision versions
        model = models.squeezenet1_1(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, NUM_CLASSES, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # Training storage
    # -------------------------
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    best_val_acc = 0.0
    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / train_len
        epoch_train_acc = running_corrects / train_len
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()

        epoch_val_loss = val_running_loss / val_len
        epoch_val_acc = val_running_corrects / val_len
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)

        print(f"\nEpoch {epoch} Summary: Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}\n")

        # Save best model during training
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ New best model saved (val acc: {best_val_acc:.4f}) -> {best_model_path}")

    print("Training completed. Best val acc:", best_val_acc)

    # -------------------------
    # Plot Accuracy & Loss
    # -------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, EPOCHS+1), train_acc_list, label='Train Acc')
    plt.plot(range(1, EPOCHS+1), val_acc_list, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, EPOCHS+1), train_loss_list, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_loss_list, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Final evaluation on test set
    # -------------------------
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    final_acc = (y_pred == y_true).mean() * 100.0
    print(f"\n🔥 FINAL ACCURACY: {final_acc:.2f} %")

    # Precision / Recall / F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    # Plot confusion matrix raw and normalized
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=plt.gca(), cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix (Raw)")

    plt.subplot(1,2,2)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(ax=plt.gca(), cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.show()

    # Bar graph for precision/recall/f1
    plt.figure(figsize=(10,5))
    x = np.arange(len(class_names))
    width = 0.25
    plt.bar(x - width, precision, width=width, label='Precision')
    plt.bar(x, recall, width=width, label='Recall')
    plt.bar(x + width, f1, width=width, label='F1')
    plt.xticks(x, class_names, rotation=45)
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 per class")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Save final model as TFLite after all epochs and validation
    # -------------------------
    print("\n=== Starting export: PyTorch -> ONNX -> SavedModel -> TFLite ===")
    print("Loading best model for TFLite export...")

    # Load best model for export
    try:
        from torchvision.models import SqueezeNet1_1_Weights
        export_model = models.squeezenet1_1(weights=None)
    except (TypeError, AttributeError):
        # Fallback for older torchvision versions
        export_model = models.squeezenet1_1(pretrained=False)
    
    export_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, NUM_CLASSES, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    )
    export_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    export_model.eval()

    # ONNX export (temporary, needed for TFLite conversion)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"✓ ONNX model exported to: {onnx_path}")
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            print("\n❌ Error: 'onnxscript' module is required for ONNX export.")
            print("Please install it using: pip install onnxscript")
            print("Or: conda install -c conda-forge onnxscript")
            raise
        else:
            print("Failed to export ONNX model:", e)
            raise
    except Exception as e:
        print("Failed to export ONNX model:", e)
        raise

    # Convert ONNX -> TensorFlow SavedModel using onnx-tf
    try:
        import onnx
        from onnx_tf.backend import prepare
        print("Loading ONNX model for conversion...")
        onnx_model = onnx.load(onnx_path)
        print("Preparing TF representation with onnx-tf (this may take a while)...")
        tf_rep = prepare(onnx_model)
        # export_graph will produce the SavedModel at saved_model_dir
        tf_rep.export_graph(saved_model_dir)
        print(f"✓ SavedModel exported to: {saved_model_dir}")
    except Exception as e:
        print("ONNX -> SavedModel conversion failed. Error:", e)
        print("Common reasons: onnx-tf / onnx / tensorflow version incompatibility.")
        raise

    # Convert SavedModel -> TFLite (Final output)
    try:
        import tensorflow as tf
        print("Converting SavedModel to TFLite (may take a while)...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        # Uncomment optimizations if you want to enable default optimizations
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved to: {tflite_path}")
    except Exception as e:
        print("SavedModel -> TFLite conversion failed. Error:", e)
        raise

    print("\n=== EXPORT COMPLETE ===")
    print(f"Final TFLite model saved to: {tflite_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
