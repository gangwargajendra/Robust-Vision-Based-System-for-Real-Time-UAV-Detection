#!/bin/bash

#######################################
# YOLO11 Resume Training Launcher
# Continues from previous checkpoint up to epoch 500
#######################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}OK: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARN: $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECOND_RESULTS_DIR="$ROOT_DIR/second/results"

print_info "=========================================="
print_info "YOLO11 Resume Training Launcher"
print_info "=========================================="
print_info "Script directory: $SCRIPT_DIR"

mkdir -p "$SECOND_RESULTS_DIR"

if ! command -v python3 &> /dev/null; then
    print_error "python3 not found"
    exit 1
fi

FULL_LOG="$SECOND_RESULTS_DIR/full_terminal_resume_201_to_500.txt"
print_info "Writing terminal output to: $FULL_LOG"

GPU_INDEX="0"
if command -v nvidia-smi &> /dev/null; then
    GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -n1 | awk -F',' '{gsub(/ /, "", $1); print $1}')
fi

print_info "Using GPU index: $GPU_INDEX"
print_info "Starting resume training (target epoch: 500)"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 "$SCRIPT_DIR/train_model_resume_201_to_500.py" >> "$FULL_LOG" 2>&1

RESULT=$?
if [ $RESULT -eq 0 ]; then
    print_success "Resume training completed successfully"
else
    print_error "Resume training failed with exit code $RESULT"
    print_info "Check log: $FULL_LOG"
    exit 1
fi

print_info "=========================================="
print_success "All done"
print_info "=========================================="
