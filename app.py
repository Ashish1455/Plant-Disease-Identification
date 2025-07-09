"""
Plant Disease Classification App
This app uses TensorFlow Lite models to predict plant diseases from images.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import argparse
from pathlib import Path


class PlantDiseaseClassifier:
    def __init__(self, model_path, class_names_path=None):
        """
        Initialize the plant disease classifier.

        Args:
            model_path (str): Path to the TensorFlow Lite model file
            class_names_path (str): Path to the JSON file containing class names
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = None

        # Load the TFLite model
        self.load_model()

        # Load class names if provided
        if class_names_path and os.path.exists(class_names_path):
            self.load_class_names(class_names_path)

    def load_model(self):
        """Load the TensorFlow Lite model."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"Model loaded successfully from {self.model_path}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_class_names(self, class_names_path):
        """Load class names from JSON file."""
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded {len(self.class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")

    def preprocess_image(self, image_path):
        """
        Preprocess the input image for model prediction.

        Args:
            image_path (str): Path to the input image

        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Get input shape from model
            input_shape = self.input_details[0]['shape']
            img_height, img_width = input_shape[1], input_shape[2]

            # Load and preprocess image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((img_width, img_height))

            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            return image_array

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise

    def predict(self, image_path):
        """
        Make prediction on the input image.

        Args:
            image_path (str): Path to the input image

        Returns:
            dict: Prediction results containing class, confidence, and probabilities
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            probabilities = output[0]

            # Get prediction results
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]

            # Format results
            results = {
                'predicted_class_index': int(predicted_class_idx),
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            }

            # Add class name if available
            if self.class_names:
                results['predicted_class'] = self.class_names[predicted_class_idx]
                results['all_classes'] = {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }

            return results

        except Exception as e:
            print(f"Error making prediction: {e}")
            raise

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images.

        Args:
            image_paths (list): List of paths to input images

        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        return results


def create_class_names_file(class_names, output_path):
    """
    Create a JSON file with class names.

    Args:
        class_names (list): List of class names
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Classification App")
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the TensorFlow Lite model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the input image or directory of images')
    parser.add_argument('--classes', type=str, default=None,
                       help='Path to the JSON file containing class names')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Path to save prediction results')

    args = parser.parse_args()

    # Initialize classifier
    classifier = PlantDiseaseClassifier(args.model, args.classes)

    # Check if input is a single image or directory
    if os.path.isfile(args.image):
        # Single image prediction
        print(f"Making prediction for: {args.image}")
        result = classifier.predict(args.image)

        print("\nPrediction Results:")
        print(f"Image: {args.image}")
        if 'predicted_class' in result:
            print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")

        if 'all_classes' in result:
            print("\nTop 5 predictions:")
            sorted_classes = sorted(result['all_classes'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_classes[:5]:
                print(f"  {class_name}: {prob:.4f}")

        # Save results
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

    elif os.path.isdir(args.image):
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(args.image).glob(f'*{ext}'))
            image_paths.extend(Path(args.image).glob(f'*{ext.upper()}'))

        if not image_paths:
            print(f"No images found in directory: {args.image}")
            return

        print(f"Found {len(image_paths)} images")
        results = classifier.predict_batch([str(path) for path in image_paths])

        # Display results
        print("\nBatch Prediction Results:")
        for result in results:
            if 'error' not in result:
                print(f"\n{result['image_path']}:")
                if 'predicted_class' in result:
                    print(f"  Predicted: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.4f}")
            else:
                print(f"\n{result['image_path']}: ERROR - {result['error']}")

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    else:
        print(f"Invalid input path: {args.image}")
        return

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
