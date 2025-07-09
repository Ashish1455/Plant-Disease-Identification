#!/usr/bin/env python3
"""
Simple setup script for Plant Disease Classification GUI
This script helps set up the project environment and creates necessary directories.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary project directories"""
    directories = [
        'models',
        'tflite_models',
        'datasets',
        'sample_images',
        'results'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def test_tkinter():
    """Test if tkinter is available"""
    try:
        import tkinter
        print("✅ tkinter is available")
        return True
    except ImportError:
        print("❌ tkinter is not available. Please install tkinter.")
        return False

def main():
    print("🌱 Plant Disease Classification - Setup Script")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check tkinter
    if not test_tkinter():
        sys.exit(1)

    # Create directories
    print("\nCreating project directories...")
    create_directories()

    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies():
        sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train your model: python train_multiple_model.py --epoch 100 --batch_size 32 --root_dir ./datasets --img_size 224 --h5_dir ./models")
    print("2. Convert to TensorFlow Lite: python convert_tflite.py --models_dir ./models --output ./tflite_models --quantize")
    print("3. Launch the GUI: python app.py")

if __name__ == "__main__":
    main()
