"""
Setup Script for YOLO11 Training Environment
Checks dependencies and installs required packages
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n>>> {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False


def is_package_installed(module_name):
    """Return True if a module import succeeds."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def setup_environment():
    """Setup training environment"""
    print("\n" + "="*70)
    print("YOLO11 TRAINING ENVIRONMENT SETUP")
    print("="*70)
    
    # Check Python version
    print(f"\nPython Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    if sys.version_info >= (3, 13):
        print("⚠️  Python 3.13 can be unstable for ML wheels on some servers.")
        print("   Recommended: conda create -n yolo11 python=3.10 -y")
        return False
    
    print("\n>>> Installing/Upgrading required packages...")
    
    packages = [
        "ultralytics>=8.0.0",
        "pyyaml",
        "opencv-python",
        "pillow",
        "numpy",
        "scipy",
        "matplotlib",
    ]
    
    for package in packages:
        print(f"\n  Installing {package}...")
        command = f"{sys.executable} -m pip install {package}"
        run_command(command, f"Installing {package}")

    # Torch is often pre-installed in GPU envs. Only install if missing.
    if not is_package_installed("torch"):
        print("\n  Installing torch and torchvision (not found in current env)...")
        run_command(f"{sys.executable} -m pip install torch torchvision", "Installing torch + torchvision")
    
    print("\n" + "="*70)
    print("✓ Environment setup completed!")
    print("="*70)
    
    # Test imports
    print("\n>>> Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✓ YOLO11 library imported successfully")
    except ImportError as e:
        print(f"❌ YOLO import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
