"""
Guava Leaf Disease Classification using ShuffleNetV2 (PyTorch)

Dataset layout (single root directory):
guava_dataset/
  healthy/
  leaf_blight/
  leaf_miner/
  leaf_spot/
  nutritional disorder/

This script:
- Loads all data from a single root with ImageFolder
- Splits into train/val in code (no file copying), e.g. 80% / 20%
- Trains ShuffleNetV2
- Plots training & validation loss/accuracy
- Computes confusion matrix (raw + normalized)
- Prints precision, recall, F1-score per class
- Prints final accuracy as: Final accuracy : x.xx%
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import datasets, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

# =======================
# 1. ShuffleNetV2 Model
# =======================


def channel_shuffle(x, groups: int):
    N, C, H, W = x.size()
    if groups == 1:
        return x
    assert C % groups == 0, 'C must be divisible by groups'
    channels_per_group = C // groups
    x = x.view(N, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(N, C, H, W)


class DepthwiseConv2d(nn.Module):
    def _init_(self, in_channels, kernel_size, stride=1, padding=0):
        super()._init_()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class ShuffleUnit(nn.Module):
    def _init_(self, in_channels, out_channels, stride):
        super()._init_()
        assert stride in [1, 2]
        self.stride = stride
        mid_channels = out_channels // 2

        if self.stride == 1:
            assert in_channels == out_channels
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                DepthwiseConv2d(mid_channels, kernel_size=3,
                                stride=1, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                DepthwiseConv2d(in_channels, kernel_size=3,
                                stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                DepthwiseConv2d(mid_channels, kernel_size=3,
                                stride=2, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            c = x.shape[1]
            assert c % 2 == 0
            x1, x2 = torch.split(x, c // 2, dim=1)
            out2 = self.branch2(x2)
            out = torch.cat((x1, out2), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def _init_(self, stages_repeats=[4, 8, 4], width_mult=1.0, num_classes=5):
        super()._init_()

        if width_mult == 0.5:
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError('Unsupported width_mult: {}'.format(width_mult))

        input_channels = 3
        output_channels = stages_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_names = []
        for idx, (repeats, output_channels) in enumerate(
            zip(stages_repeats, stages_out_channels[1:4])
        ):
            stage = self._make_stage(input_channels, output_channels, repeats)
            stage_name = f"stage{idx + 2}"
            setattr(self, stage_name, stage)
            self.stage_names.append(stage_name)
            input_channels = output_channels

        last_output = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, last_output, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_output),
            nn.ReLU(inplace=True),
        )

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_output, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = [ShuffleUnit(in_channels, out_channels, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def shufflenet_v2_x1_0(num_classes=5):
    return ShuffleNetV2(width_mult=1.0, num_classes=num_classes)


# =======================
# 2. Dataset & In-code Split
# =======================

# TODO: set this to your dataset root folder (single root with 5 class folders)
DATA_DIR = r"C:\\Users\\shush\\Downloads\\augmented\\augmented"  # <-- CHANGE THIS

input_size = 224

transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Load the full dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(full_dataset))

# We know you have balanced classes (2000 each), but let's make it generic:


def stratified_split(dataset, val_ratio=0.2, shuffle=True, random_seed=42):
    """
    Split indices into train and val in a stratified way (per class).
    Does NOT move/copy files, only returns index lists.
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = []
    val_indices = []

    for cls in np.unique(targets):
        cls_indices = indices[targets[indices] == cls]
        n_total = len(cls_indices)
        n_val = int(math.floor(val_ratio * n_total))
        val_idx = cls_indices[:n_val]
        train_idx = cls_indices[n_val:]
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    return train_indices, val_indices


train_indices, val_indices = stratified_split(full_dataset, val_ratio=0.2)

print("Train size:", len(train_indices))
print("Val size:", len(val_indices))

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

batch_size = 32

train_loader = DataLoader(
    full_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    full_dataset,
    batch_size=batch_size,
    sampler=val_sampler,
    num_workers=4,
    pin_memory=True,
)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_indices), "val": len(val_indices)}


# =======================
# 3. Training Utilities
# =======================

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print("-" * 40)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("-" * 40)
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    return model, history


def plot_training_curves(history, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(history["train_losses"], label="Train Loss")
    plt.plot(history["val_losses"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history["train_accuracies"], label="Train Accuracy")
    plt.plot(history["val_accuracies"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    print(
        f"Saved training curves to '{out_dir}/loss_curve.png' and '{out_dir}/accuracy_curve.png'")


def plot_confusion_matrices(cm, class_names, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # Raw confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_dir, "confusion_matrix_normalized.png"), dpi=300)
    plt.close()

    print(
        f"Saved confusion matrices to '{out_dir}/confusion_matrix.png' and "
        f"'{out_dir}/confusion_matrix_normalized.png'"
    )


def evaluate_model(model, dataloader, device, class_names, out_dir="plots"):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrices(cm, class_names, out_dir=out_dir)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )
    acc = accuracy_score(all_labels, all_preds)

    print("\nClassification Report (per class):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    for idx, cls_name in enumerate(class_names):
        print(
            f"Class: {cls_name:20s} | "
            f"Precision: {precision[idx]:.4f} | "
            f"Recall: {recall[idx]:.4f} | "
            f"F1-score: {f1[idx]:.4f}"
        )

    print(f"\nFinal accuracy : {acc * 100:.2f}%")

    return acc, precision, recall, f1


# =======================
# 4. Main
# =======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = shufflenet_v2_x1_0(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # TODO: set epochs as you like
    NUM_EPOCHS = 30  # <-- CHANGE IF YOU WANT

    best_model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_model.state_dict(),
               "checkpoints/guava_shufflenetv2_best.pth")
    print("Model saved to checkpoints/guava_shufflenetv2_best.pth")

    plot_training_curves(history, out_dir="plots")

    evaluate_model(
        best_model, dataloaders["val"], device, class_names, out_dir="plots")


if _name_ == "_main_":
    main()