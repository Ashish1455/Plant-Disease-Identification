# Plant Disease Classification

A comprehensive machine learning project for classifying plant diseases from images using TensorFlow and deep learning. This project utilizes EfficientNetB0 architecture for accurate disease detection and provides both training capabilities and a user-friendly inference application.

## 🌱 Features

- **Multi-class Plant Disease Classification**: Supports classification of 38 different plant disease categories
- **EfficientNetB0 Backbone**: Utilizes state-of-the-art EfficientNetB0 architecture for robust feature extraction
- **TensorFlow Lite Conversion**: Optimized models for mobile and edge deployment
- **Data Augmentation**: Built-in augmentation techniques for improved model generalization
- **Batch Processing**: Support for processing multiple images simultaneously
- **Flexible Architecture**: Easy to extend for new plant diseases and datasets
- **Comprehensive Utilities**: Complete pipeline from data preparation to model deployment

## 📁 Project Structure

```
├── app.py                      # Main inference application
├── convert_tflite.py          # Keras to TensorFlow Lite conversion
├── model.py                   # Model architecture definition
├── opt.py                     # Training argument parser
├── prepare_data.py            # Data preparation utilities
├── requirements.txt           # Python dependencies
├── setup.py                   # Project setup script
├── train_multiple_model.py    # Training script for multiple models
├── models/                    # Saved Keras models (.h5)
├── tflite_models/            # TensorFlow Lite models
├── datasets/                 # Training datasets
├── sample_images/            # Sample images for testing
└── results/                  # Prediction results
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ashish1455/Plant-Disease-Identification.git
   cd plant-disease-classification
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will:
   - Check Python version compatibility
   - Install required dependencies
   - Create necessary directories

3. **Download the dataset**
   
   Download the New Plant Diseases Dataset from Kaggle:
   **https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset**
   
   This dataset contains:
   - 87,000+ RGB images of healthy and diseased crop leaves
   - 38 different disease categories
   - 80/20 training/validation split
   - Additional test set for evaluation

   Extract the dataset to the `datasets/` directory.

### Manual Installation

If you prefer manual installation:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models tflite_models datasets sample_images results
```

## 🎯 Usage

### Training Models

Train disease classification models on your dataset:

```bash
python train_multiple_model.py \
    --root_dir ./datasets \
    --img_size 224 \
    --batch_size 32 \
    --epoch 100 \
    --h5_dir ./models
```

**Parameters:**
- `--root_dir`: Path to dataset directory
- `--img_size`: Input image size (recommended: 224)
- `--batch_size`: Training batch size
- `--epoch`: Number of training epochs
- `--h5_dir`: Directory to save trained models

### Converting to TensorFlow Lite

Convert trained Keras models to TensorFlow Lite for optimized inference:

```bash
# Convert single model
python convert_tflite.py \
    --model ./models/model.h5 \
    --output ./tflite_models/model.tflite \
    --quantize

# Convert multiple models
python convert_tflite.py \
    --models_dir ./models \
    --output ./tflite_models \
    --quantize

# Extract class names
python convert_tflite.py \
    --model ./models/model.h5 \
    --output ./tflite_models/model.tflite \
    --dataset ./datasets \
    --classes_output class_names.json
```

### Running Inference

Use the inference application to classify plant diseases:

```bash
# Single image prediction
python app.py \
    --model ./tflite_models/model.tflite \
    --image ./sample_images/leaf.jpg \
    --classes class_names.json

# Batch prediction
python app.py \
    --model ./tflite_models/model.tflite \
    --image ./sample_images/ \
    --classes class_names.json \
    --output batch_results.json
```

## 📊 Model Architecture

The project uses **EfficientNetB0** as the backbone architecture with the following enhancements:

- **Input Layer**: Flexible input shape (default: 224x224x3)
- **Backbone**: EfficientNetB0 (ImageNet pretrained)
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layers**: 512 → 256 → num_classes
- **Regularization**: L2 regularization and Dropout (0.2)
- **Activation**: ReLU for hidden layers, Softmax for output

### Data Augmentation

Built-in augmentation techniques include:
- Random rotation (±15°)
- Random translation (±10%)
- Random horizontal/vertical flipping
- Random contrast adjustment (±10%)

## 🔧 Configuration

### Training Parameters

Modify training parameters in `opt.py` or use command-line arguments:

```python
# Default parameters
--img_size 224        # Input image size
--batch_size 32       # Training batch size
--epoch 100          # Number of epochs
--root_dir ./datasets # Dataset directory
```

### Model Configuration

Customize model architecture in `model.py`:

```python
# Modify dense layer sizes
x = layers.Dense(512, activation='relu')(x)  # First dense layer
x = layers.Dense(256, activation='relu')(x)  # Second dense layer

# Adjust dropout rate
x = layers.Dropout(0.2)(x)  # Dropout rate
```

## 📈 Results and Performance

The model achieves high accuracy on the plant disease classification task:

- **Training Accuracy**: ~95%+ (depends on dataset and training parameters)
- **Validation Accuracy**: ~90%+ (with proper regularization)
- **Inference Speed**: Fast inference with TensorFlow Lite optimization
- **Model Size**: Significantly reduced with quantization

## 🛠️ Dependencies

Core dependencies:
- `tensorflow>=2.17.0`
- `numpy>=1.26.4`
- `pillow>=10.2.0`
- `keras>=3.3.3`
- `opencv-python>=4.9.0.80`
- `tqdm>=4.66.4`

See `requirements.txt` for complete dependency list.

## 🗂️ Dataset Information

### New Plant Diseases Dataset

- **Source**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Size**: ~87,000 images (1.43 GB)
- **Classes**: 38 different plant disease categories
- **Format**: RGB images of crop leaves
- **Split**: 80% training, 20% validation + test set

