# ✅ Flickr8k Dataset Setup Complete!

**Date:** November 8, 2025  
**Status:** READY FOR TRAINING ✅

---

## Problem Solved

You encountered:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
```

## What Was Done

### 1. ✅ Created Symlink
```bash
ln -s data/flickr8k flickr8k
```
The dataset was in `data/flickr8k` but training scripts expected `./flickr8k`

### 2. ✅ Extracted Zip Files
```bash
cd data/flickr8k
unzip Flickr8k_Dataset.zip
unzip Flickr8k_text.zip
```

### 3. ✅ Fixed Directory Structure
```bash
# Renamed Flicker8k_Dataset → Flickr8k_Dataset (fixed typo)
mv Flicker8k_Dataset Flickr8k_Dataset

# Created Flickr8k_text directory
mkdir Flickr8k_text

# Moved text files to proper location
mv Flickr_8k.*.txt Flickr8k_text/
mv Flickr8k.token.txt Flickr8k_text/
```

### 4. ✅ Fixed Training Script
Changed `batch['images']` → `batch['image']` to match dataset output keys

---

## Current Directory Structure

```
flickr8k/ (symlink → data/flickr8k)
├── Flickr8k_Dataset/           # 8,000 images
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d70.jpg
│   └── ... (8,000 total)
├── Flickr8k_text/              # Text annotations
│   ├── Flickr_8k.trainImages.txt   (6,000 images)
│   ├── Flickr_8k.devImages.txt     (1,000 images)
│   ├── Flickr_8k.testImages.txt    (1,000 images)
│   └── Flickr8k.token.txt          (40,000 captions)
└── (audio files optional - will use dummy audio if not present)
```

---

## Dataset Stats

### Dataset Loaded Successfully! ✅

```
✓ Train split: 30,000 samples (6,000 images × 5 captions each)
✓ Val split:   5,000 samples (1,000 images × 5 captions each)
✓ Test split:  5,000 samples (1,000 images × 5 captions each)

Sample structure:
  - image: torch.Size([3, 224, 224])  # RGB image
  - text: torch.Size([77])             # Token IDs
  - audio: torch.Size([80, 100])       # Mel spectrogram (dummy if no audio files)
```

---

## Ready to Train! 🚀

### Quick Start:

```bash
# Basic training (recommended)
python train_flickr8k.py --data_dir ./flickr8k --epochs 30

# Or use core nl_mm
python train_nlmm_flickr8k.py \
    --config modules/nl_mm/configs/tiny_single_gpu.yaml \
    --data_dir ./flickr8k \
    --epochs 30 \
    --batch_size 32
```

### What Will Happen:

1. ✅ Loads 30,000 training samples (image + text pairs)
2. ✅ Uses dummy audio (zeros) since FACC audio files aren't downloaded
3. ✅ Trains with contrastive learning (InfoNCE)
4. ✅ Evaluates cross-modal retrieval (image↔text)
5. ✅ Saves best model based on R@1 metrics

---

## About Audio Data

### Current Status: Image + Text Only ⚠️

The Flickr8k images and text captions are ready. However:

**Audio files (FACC - Flickr Audio Caption Corpus) are optional:**
- Not included in standard Flickr8k dataset
- Requires separate download (~2GB)
- Dataset loader will use dummy audio (zeros) if not present
- Training will still work but without real audio modality

### If You Want Real Audio:

The FACC dataset provides spoken versions of the captions.

**Download from:**
- Official: https://groups.csail.mit.edu/sls/downloads/flickraudio/
- Requires registration and academic use agreement

**After downloading:**
```bash
cd data/flickr8k
mkdir -p flickr_audio/wavs
# Extract audio files to flickr_audio/wavs/
# Add wav2capt.txt mapping file to flickr_audio/
```

**For now:** Training works fine with image+text only! Audio is optional.

---

## Training Options

### Option 1: Image + Text (Current Setup) ✅
```bash
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```
- Uses real images and text
- Uses dummy audio (no impact on image↔text training)
- **Best for:** Getting started, image captioning

### Option 2: Full Tri-Modal (Need FACC)
```bash
# After downloading FACC audio files
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```
- Uses real images, text, AND audio
- True tri-modal learning
- **Best for:** Complete multimodal research

---

## Troubleshooting

### Issue: "FileNotFoundError" again
**Solution:** The symlink and structure are now correct. If error persists:
```bash
cd /home/alex/Documents/DeepLearning2
ls -la flickr8k/Flickr8k_Dataset/
ls -la flickr8k/Flickr8k_text/
```

### Issue: "KeyError: 'images'"
**Solution:** ✅ Already fixed in train_nlmm_flickr8k.py

### Issue: Slow training
**Solution:**
```bash
# Reduce batch size or image size
python train_flickr8k.py --batch_size 16 --image_size 128
```

### Issue: Out of memory
**Solution:**
```bash
# Use smaller config
python train_nlmm_flickr8k.py \
    --config modules/nl_mm/configs/tiny_single_gpu.yaml \
    --batch_size 16
```

---

## Next Steps

### 1. Start Training (Now!) 🚀
```bash
python train_flickr8k.py --data_dir ./flickr8k --epochs 30
```

### 2. Monitor Progress
```bash
# In another terminal
tail -f flickr8k_training.log
watch -n 1 nvidia-smi
```

### 3. Evaluate Results
```bash
python show_flickr8k_results.py
python test_multimodal_trained.py --checkpoint results/folder_per_model/multimodal_memory/outputs/best_model.pt
```

### 4. (Optional) Add Audio Later
If you want to train with real audio:
1. Download FACC dataset
2. Extract to `flickr_audio/wavs/`
3. Re-run training with same commands

---

## Summary

### Before:
```
❌ FileNotFoundError: flickr8k directory missing
❌ Zip files not extracted
❌ Wrong directory structure
❌ Training script key mismatch
```

### After:
```
✅ Symlink created: flickr8k → data/flickr8k
✅ 8,000 images extracted and accessible
✅ 40,000 text captions loaded
✅ Directory structure matches dataset loader
✅ Training script fixed
✅ 30,000 training samples ready
✅ READY TO TRAIN!
```

---

## Quick Test

Verify everything works:
```bash
python3 -c "
from utils.flickr8k_dataset import Flickr8kAudioDataset
ds = Flickr8kAudioDataset('./flickr8k', 'train', image_size=224)
print(f'✅ Dataset loaded: {len(ds)} samples')
sample = ds[0]
print(f'✅ Sample loaded: image {sample[\"image\"].shape}, text {sample[\"text\"].shape}')
print('✅ Ready to train!')
"
```

**Expected output:**
```
✅ Dataset loaded: 30000 samples
✅ Sample loaded: image torch.Size([3, 224, 224]), text torch.Size([77])
✅ Ready to train!
```

---

**Setup Date:** November 8, 2025  
**Dataset:** Flickr8k (images + text)  
**Status:** ✅ READY FOR TRAINING  
**Action:** Run training command! 🚀

