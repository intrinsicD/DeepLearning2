#!/bin/bash
# Training wrapper for NL-MM model
# Usage: ./train_nlmm.sh [additional arguments]

# Set PYTHONPATH to include project root
export PYTHONPATH="/home/alex/Documents/DeepLearning2:$PYTHONPATH"

# Change to project directory
cd /home/alex/Documents/DeepLearning2

# Run training script with all arguments passed through
python training/scripts/train_nlmm_flickr8k.py "$@"

