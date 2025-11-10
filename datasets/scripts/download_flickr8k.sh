#!/bin/bash
# Download Flickr8k dataset for multimodal training
# This script downloads images and text captions

set -e

echo "============================================================"
echo "Flickr8k Dataset Download Script"
echo "============================================================"
echo ""

# Create directory structure
FLICKR_DIR="./data/flickr8k"
mkdir -p "$FLICKR_DIR"
cd "$FLICKR_DIR"

echo "Target directory: $(pwd)"
echo ""

# Check if already downloaded
if [ -d "Flickr8k_Dataset" ] && [ -d "Flickr8k_text" ]; then
    echo "✓ Flickr8k dataset already exists!"
    echo "  - Images: Flickr8k_Dataset/"
    echo "  - Text: Flickr8k_text/"

    # Count images
    NUM_IMAGES=$(find Flickr8k_Dataset -name "*.jpg" | wc -l)
    echo "  - Found $NUM_IMAGES images"

    echo ""
    echo "If you want to re-download, delete these directories first."
    exit 0
fi

echo "============================================================"
echo "Step 1: Downloading Flickr8k Images"
echo "============================================================"
echo ""

# Download images
if [ ! -f "Flickr8k_Dataset.zip" ]; then
    echo "Downloading Flickr8k images (1GB)..."
    wget --no-check-certificate https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    echo "✓ Download complete"
else
    echo "✓ Flickr8k_Dataset.zip already exists"
fi

echo ""
echo "Extracting images..."
unzip -q Flickr8k_Dataset.zip
echo "✓ Extraction complete"

# Count images
NUM_IMAGES=$(find Flickr8k_Dataset -name "*.jpg" | wc -l)
echo "✓ Found $NUM_IMAGES images"

echo ""
echo "============================================================"
echo "Step 2: Downloading Flickr8k Text Annotations"
echo "============================================================"
echo ""

# Download text
if [ ! -f "Flickr8k_text.zip" ]; then
    echo "Downloading Flickr8k text annotations..."
    wget --no-check-certificate https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
    echo "✓ Download complete"
else
    echo "✓ Flickr8k_text.zip already exists"
fi

echo ""
echo "Extracting text annotations..."
unzip -q Flickr8k_text.zip
echo "✓ Extraction complete"

# Check files
echo ""
echo "Verifying text files:"
for file in Flickr8k.token.txt Flickr_8k.trainImages.txt Flickr_8k.devImages.txt Flickr_8k.testImages.txt; do
    if [ -f "Flickr8k_text/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
    fi
done

echo ""
echo "============================================================"
echo "Step 3: Directory Structure"
echo "============================================================"
echo ""
tree -L 2 -h || ls -lh

echo ""
echo "============================================================"
echo "FLICKR8K DOWNLOAD COMPLETE!"
echo "============================================================"
echo ""
echo "Dataset location: $FLICKR_DIR"
echo "  - Images: $(find Flickr8k_Dataset -name "*.jpg" | wc -l) JPG files"
echo "  - Text: Flickr8k_text/"
echo ""
echo "Note: Flickr8k Audio Caption Corpus (FACC) is optional."
echo "      The training script can work with just images + text."
echo ""
echo "To download FACC audio (optional, 40K audio files, ~3GB):"
echo "  Visit: https://www.kaggle.com/datasets/warcoder/flickr-8k-audio-caption-corpus"
echo "  Or:    https://sls.csail.mit.edu/downloads/flickraudio/"
echo ""
echo "Ready to train!"
echo "  python train_flickr8k.py --data_dir ./data/flickr8k"
echo ""

