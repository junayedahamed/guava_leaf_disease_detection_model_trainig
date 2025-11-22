# """
# Convert trained PyTorch .pth model to TFLite format.

# This script loads a saved .pth model and converts it to TFLite format
# without needing to retrain the model.

# Usage:
#     python convert_pth_to_tflite.py
# """

# import os
# import shutil
# import torch
# import torch.nn as nn
# from torchvision import models

# # -------------------------
# # CONFIGURATION
# # -------------------------
# # Path to your trained .pth model
# PTH_MODEL_PATH = r"D:\model_trainig\squeezenet\trained_model\best_model.pth"

# # Model architecture parameters (must match training script)
# NUM_CLASSES = 5
# IMG_SIZE = 224

# # Output paths
# OUTPUT_DIR = r"D:\model_trainig\squeezenet\trained_model"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ONNX_PATH = os.path.join(OUTPUT_DIR, "onnx_model.onnx")
# SAVED_MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_model")
# TFLITE_PATH = os.path.join(OUTPUT_DIR, "squeezenet_model.tflite")

# print("=" * 60)
# print("PyTorch (.pth) to TFLite Converter")
# print("=" * 60)
# print(f"📁 Input model: {PTH_MODEL_PATH}")
# print(f"📁 Output directory: {OUTPUT_DIR}")
# print(f"📊 Number of classes: {NUM_CLASSES}")
# print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")
# print("=" * 60)

# # -------------------------
# # Load and recreate model architecture
# # -------------------------
# print("\n[1/4] Loading model architecture...")

# try:
#     from torchvision.models import SqueezeNet1_1_Weights
#     model = models.squeezenet1_1(weights=None)
# except (TypeError, AttributeError):
#     # Fallback for older torchvision versions
#     model = models.squeezenet1_1(pretrained=False)

# # Recreate the classifier (must match training script)
# model.classifier = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Conv2d(512, NUM_CLASSES, kernel_size=1),
#     nn.ReLU(inplace=True),
#     nn.AdaptiveAvgPool2d((1, 1))
# )

# # Load trained weights
# if not os.path.exists(PTH_MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {PTH_MODEL_PATH}")

# print(f"Loading weights from: {PTH_MODEL_PATH}")
# model.load_state_dict(torch.load(PTH_MODEL_PATH, map_location='cpu'))
# model.eval()
# print("✓ Model loaded successfully")

# # -------------------------
# # Export to ONNX
# # -------------------------
# print("\n[2/4] Converting to ONNX format...")
# dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# try:
#     # Use opset 18 (newer default) to avoid version conversion issues
#     torch.onnx.export(
#         model,
#         dummy_input,
#         ONNX_PATH,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=18,  # Updated to 18 for better compatibility
#         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
#     )
#     print(f"✓ ONNX model exported to: {ONNX_PATH}")
# except ModuleNotFoundError as e:
#     if "onnxscript" in str(e):
#         print("\n❌ Error: 'onnxscript' module is required for ONNX export.")
#         print("Please install it using: pip install onnxscript")
#         raise
#     else:
#         print(f"❌ Failed to export ONNX model: {e}")
#         raise
# except Exception as e:
#     print(f"⚠️  Warning during ONNX export: {e}")
#     # Check if file was still created despite warnings
#     if os.path.exists(ONNX_PATH):
#         print(f"✓ ONNX model was created despite warnings: {ONNX_PATH}")
#     else:
#         print(f"❌ Failed to export ONNX model: {e}")
#         raise

# # -------------------------
# # Convert ONNX to TensorFlow SavedModel
# # -------------------------
# print("\n[3/4] Converting ONNX to TensorFlow SavedModel...")

# # Remove existing saved_model directory if it exists
# if os.path.exists(SAVED_MODEL_DIR):
#     shutil.rmtree(SAVED_MODEL_DIR)

# # Try onnx2tf first (newer, more compatible)
# try:
#     print("Trying onnx2tf (recommended)...")
#     import onnx2tf
#     onnx2tf.convert(
#         input_onnx_file_path=ONNX_PATH,
#         output_folder_path=SAVED_MODEL_DIR,
#         copy_onnx_input_output_names_to_tflite=True
#     )
#     print(f"✓ SavedModel exported to: {SAVED_MODEL_DIR}")
# except ImportError as e:
#     print(f"onnx2tf import failed: {e}")
#     print("Trying onnx-tf (legacy, may not work with newer ONNX versions)...")
#     try:
#         import onnx
#         from onnx_tf.backend import prepare
#         print("Loading ONNX model...")
#         onnx_model = onnx.load(ONNX_PATH)
#         print("Converting to TensorFlow format (this may take a while)...")
#         tf_rep = prepare(onnx_model)
#         tf_rep.export_graph(SAVED_MODEL_DIR)
#         print(f"✓ SavedModel exported to: {SAVED_MODEL_DIR}")
#     except ImportError as e2:
#         print(f"\n❌ Error: Required library not found.")
#         print("Please install: pip install onnx2tf tf_keras onnx-graphsurgeon")
#         print("Note: onnx-tf is not compatible with newer ONNX versions")
#         raise
#     except Exception as e2:
#         print(f"❌ ONNX -> SavedModel conversion failed with onnx-tf: {e2}")
#         print("\nNote: onnx-tf is not compatible with newer ONNX versions.")
#         print("Please install: pip install onnx2tf tf_keras onnx-graphsurgeon")
#         raise
# except Exception as e:
#     print(f"❌ ONNX -> SavedModel conversion failed with onnx2tf: {e}")
#     print("\nTrying fallback method with onnx-tf (may not work)...")
#     try:
#         import onnx
#         from onnx_tf.backend import prepare
#         onnx_model = onnx.load(ONNX_PATH)
#         tf_rep = prepare(onnx_model)
#         tf_rep.export_graph(SAVED_MODEL_DIR)
#         print(f"✓ SavedModel exported to: {SAVED_MODEL_DIR}")
#     except ImportError as e2:
#         print(f"❌ onnx-tf not available or incompatible.")
#         print(f"Original onnx2tf error: {e}")
#         print("\nPlease ensure: pip install onnx2tf tf_keras onnx-graphsurgeon")
#         raise
#     except Exception as e2:
#         print(f"❌ Both conversion methods failed.")
#         print(f"onnx2tf error: {e}")
#         print(f"onnx-tf error: {e2}")
#         print("\nNote: onnx-tf is not compatible with newer ONNX versions.")
#         print("Please ensure: pip install onnx2tf tf_keras onnx-graphsurgeon")
#         raise

# # -------------------------
# # Convert SavedModel to TFLite
# # -------------------------
# print("\n[4/4] Converting SavedModel to TFLite...")

# try:
#     import tensorflow as tf
#     print("Loading SavedModel...")
#     converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
#     # Optional: Enable optimizations (uncomment if needed)
#     # converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     print("Converting to TFLite format (this may take a while)...")
#     tflite_model = converter.convert()
    
#     # Save TFLite model
#     with open(TFLITE_PATH, "wb") as f:
#         f.write(tflite_model)
    
#     # Get file size
#     file_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)  # MB
#     print(f"✓ TFLite model saved to: {TFLITE_PATH}")
#     print(f"✓ Model size: {file_size:.2f} MB")
# except ImportError as e:
#     print(f"❌ Error: TensorFlow not found: {e}")
#     print("Please install: pip install tensorflow")
#     raise
# except Exception as e:
#     print(f"❌ SavedModel -> TFLite conversion failed: {e}")
#     raise

# # -------------------------
# # Summary
# # -------------------------
# print("\n" + "=" * 60)
# print("✅ CONVERSION COMPLETE!")
# print("=" * 60)
# print(f"📁 TFLite model: {TFLITE_PATH}")
# print(f"📁 ONNX model: {ONNX_PATH}")
# print(f"📁 SavedModel: {SAVED_MODEL_DIR}")
# print("=" * 60)

# import torch
# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf
# import argparse
# import sys

# # Import your model class
# sys.path.append(".")  # make sure squeezenet_model.py is in the same folder
# from squeezenet_model import SqueezeNetModel  # adjust class name if needed

# def convert_pth_to_tflite(pth_path, tflite_path, onnx_path="model.onnx", tf_path="model_tf"):
#     # --- Step 1: Load PyTorch model ---
#     model = SqueezeNetModel()  # initialize your model
#     model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
#     model.eval()
#     print("PyTorch model loaded.")

#     # --- Step 2: Export to ONNX ---
#     dummy_input = torch.randn(1, 3, 224, 224)  # adjust input shape if different
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=11
#     )
#     print(f"ONNX model saved at: {onnx_path}")

#     # --- Step 3: Convert ONNX to TensorFlow SavedModel ---
#     onnx_model = onnx.load(onnx_path)
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_path)
#     print(f"TensorFlow SavedModel exported at: {tf_path}")

#     # --- Step 4: Convert TensorFlow SavedModel to TFLite ---
#     converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional optimization
#     tflite_model = converter.convert()

#     # Save TFLite model
#     with open(tflite_path, "wb") as f:
#         f.write(tflite_model)
#     print(f"TFLite model saved at: {tflite_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert PyTorch .pth model to TFLite")
#     parser.add_argument("--pth", type=str, required=True, help="Path to the PyTorch .pth model")
#     parser.add_argument("--tflite", type=str, required=True, help="Output path for the TFLite model")
#     parser.add_argument("--onnx", type=str, default="model.onnx", help="Intermediate ONNX path (optional)")
#     parser.add_argument("--tf", type=str, default="model_tf", help="Intermediate TensorFlow SavedModel path (optional)")
#     args = parser.parse_args()

#     convert_pth_to_tflite(args.pth, args.tflite, args.onnx, args.tf)
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import io
import sys

# Import your model class
sys.path.append(".")
from squeezenet_model import SqueezeNetModel  # adjust class name if needed

# -------------------- Paths --------------------
pth_path = r"D:\model_trainig\squeezenet\trained_model\best_model.pth"  # input .pth
tflite_path = r"D:\model_trainig\squeezenet\trained_model\best_model.tflite"  # output .tflite
# ------------------------------------------------

# --- Step 1: Load PyTorch model ---
model = SqueezeNetModel()
model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
model.eval()
print("PyTorch model loaded.")

# --- Step 2: Export to ONNX in memory ---
dummy_input = torch.randn(1, 3, 224, 224)  # adjust shape if needed
f = io.BytesIO()
torch.onnx.export(
    model,
    dummy_input,
    f,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
f.seek(0)
onnx_model = onnx.load_model(f)
print("ONNX model created in memory.")

# --- Step 3: Convert ONNX to TensorFlow in memory ---
tf_rep = prepare(onnx_model)
# tf_rep.export_graph()  # no need to export

# --- Step 4: Convert TensorFlow to TFLite ---
# Using in-memory graph
converter = tf.lite.TFLiteConverter.from_concrete_functions(tf_rep.tf_module.__call__.functions.values())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_path}")
