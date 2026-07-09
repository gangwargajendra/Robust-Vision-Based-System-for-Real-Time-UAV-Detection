#!/bin/bash

#######################################
# YOLO11 Training Launcher Script
# Run this on the GPU server
#######################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_info() {
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Main script
print_info "=========================================="
print_info "YOLO11 Training Launcher"
print_info "=========================================="

# Check current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
print_info "Working directory: $SCRIPT_DIR"

# Check if data.yaml exists
if [ ! -f "$SCRIPT_DIR/data.yaml" ]; then
    print_error "data.yaml not found in $SCRIPT_DIR"
    print_info "Please ensure data.yaml is configured properly"
    exit 1
fi

print_success "data.yaml found"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Check conda environment (if using conda)
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    print_info "Conda environment: $CONDA_DEFAULT_ENV"
else
    print_warning "Not running in a conda environment"
fi

# Optional setup only when explicitly requested
if [ "$1" = "--setup" ]; then
    if [ -f "$SCRIPT_DIR/setup_environment.py" ]; then
        print_info "Running environment setup..."
        python3 "$SCRIPT_DIR/setup_environment.py"
    fi
fi

# Select least-used GPU if nvidia-smi exists
GPU_INDEX="0"
if command -v nvidia-smi &> /dev/null; then
    GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -n1 | awk -F',' '{gsub(/ /, "", $1); print $1}')
    print_info "Selected GPU index: $GPU_INDEX (least memory used)"
fi

mkdir -p "$SCRIPT_DIR/logs"
FULL_LOG="$SCRIPT_DIR/logs/full_terminal_$(date +%Y%m%d_%H%M%S).txt"

# Run training
print_info "=========================================="
print_info "Starting YOLO11 Training"
print_info "=========================================="
print_info "Full log: $FULL_LOG"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 "$SCRIPT_DIR/train_model.py" > "$FULL_LOG" 2>&1

TRAINING_RESULT=$?

if [ $TRAINING_RESULT -eq 0 ]; then
    print_success "Training completed successfully!"
    print_info "Check logs directory for full and epoch-wise output"
else
    print_error "Training failed with exit code $TRAINING_RESULT"
    print_info "Check full log: $FULL_LOG"
    exit 1
fi

print_info "=========================================="
print_success "All done!"
print_info "=========================================="
