import torch
import torch.nn as nn
from torchvision import models
import tensorflow as tf
import shutil
import os
import subprocess

# Paths
PTH_PATH = r"D:\model_trainig\squeezenet\trained_model\best_model.pth"
TFLITE_PATH = r"D:\model_trainig\squeezenet\trained_model\best_model.tflite"

# Model config
NUM_CLASSES = 5
IMG_SIZE = 224

# Load model
model = models.squeezenet1_1(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Conv2d(512, NUM_CLASSES, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)
model.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
model.eval()

# PyTorch -> ONNX
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
onnx_path = "temp.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

# ONNX -> SavedModel (using command line to avoid import issues)
saved_model_dir = "temp_saved_model"
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)
os.makedirs(saved_model_dir, exist_ok=True)

# Use onnx2tf via command line
try:
    subprocess.run(
        ["onnx2tf", "-i", onnx_path, "-o", saved_model_dir],
        check=True,
        capture_output=True
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    # Fallback: try using onnx-tf with compatible ONNX version
    import onnx
    try:
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model_dir)
    except Exception as e:
        print(f"Error: Could not convert ONNX to SavedModel. Please install: pip install onnx-tf")
        print(f"Or ensure onnx2tf is properly installed with all dependencies.")
        raise

# SavedModel -> TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# Cleanup
os.remove(onnx_path)
shutil.rmtree(saved_model_dir)

print(f"TFLite model saved: {TFLITE_PATH}")
