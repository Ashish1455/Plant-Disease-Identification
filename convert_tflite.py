"""
Convert trained Keras models to TensorFlow Lite format for efficient inference.
"""

import tensorflow as tf
import os
import json
from pathlib import Path
import argparse


def convert_model_to_tflite(model_path, output_path, quantize=False):
    """
    Convert a Keras model to TensorFlow Lite format.

    Args:
        model_path (str): Path to the Keras model (.h5 file)
        output_path (str): Path to save the TFLite model
        quantize (bool): Whether to apply quantization for smaller model size
    """
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply quantization if requested
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print(f"Applying quantization to model: {model_path}")

        # Convert the model
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model converted successfully: {model_path} -> {output_path}")

        # Get model sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")

    except Exception as e:
        print(f"Error converting model {model_path}: {e}")


def extract_class_names(model_path, dataset_path, output_path):
    """
    Extract class names from dataset structure and save to JSON.

    Args:
        model_path (str): Path to the model (for reference)
        dataset_path (str): Path to the dataset directory
        output_path (str): Path to save class names JSON
    """
    try:
        # Look for class names in the dataset structure
        train_path = os.path.join(dataset_path, 'train')
        if os.path.exists(train_path):
            class_names = sorted(os.listdir(train_path))
            class_names = [name for name in class_names if os.path.isdir(os.path.join(train_path, name))]
        else:
            # If no train directory, look for class directories in the root
            class_names = sorted(os.listdir(dataset_path))
            class_names = [name for name in class_names if os.path.isdir(os.path.join(dataset_path, name))]

        # Save class names to JSON
        with open(output_path, 'w') as f:
            json.dump(class_names, f, indent=2)

        print(f"Extracted {len(class_names)} class names from {dataset_path}")
        print(f"Class names saved to: {output_path}")
        print(f"Classes: {class_names}")

    except Exception as e:
        print(f"Error extracting class names: {e}")


def convert_multiple_models(models_dir, output_dir, quantize=False):
    """
    Convert multiple Keras models to TensorFlow Lite format.

    Args:
        models_dir (str): Directory containing Keras models (.h5 files)
        output_dir (str): Directory to save TFLite models
        quantize (bool): Whether to apply quantization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all .h5 files
    h5_files = Path(models_dir).glob('*.h5')

    for h5_file in h5_files:
        model_name = h5_file.stem
        output_path = os.path.join(output_dir, f"{model_name}.tflite")

        convert_model_to_tflite(str(h5_file), output_path, quantize)


def main():
    parser = argparse.ArgumentParser(description="Convert Keras models to TensorFlow Lite")
    parser.add_argument('--model', type=str, help='Path to single Keras model (.h5)')
    parser.add_argument('--models_dir', type=str, help='Directory containing multiple Keras models')
    parser.add_argument('--output', type=str, help='Output path for single model or output directory for multiple models')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization for smaller model size')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory to extract class names')
    parser.add_argument('--classes_output', type=str, default='class_names.json', 
                       help='Output path for class names JSON file')

    args = parser.parse_args()

    if args.model and args.output:
        # Convert single model
        convert_model_to_tflite(args.model, args.output, args.quantize)
    elif args.models_dir and args.output:
        # Convert multiple models
        convert_multiple_models(args.models_dir, args.output, args.quantize)
    else:
        print("Please provide either --model and --output for single model conversion,")
        print("or --models_dir and --output for multiple model conversion.")
        return

    # Extract class names if dataset path is provided
    if args.dataset:
        extract_class_names(args.model or args.models_dir, args.dataset, args.classes_output)


if __name__ == "__main__":
    main()
