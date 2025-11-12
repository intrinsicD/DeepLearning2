"""Flickr8k + Flickr Audio Caption Corpus (FACC) dataset loader.

Tri-modal dataset: Images + Text captions + Spoken audio captions
- Flickr8k: 8,000 images with 5 text captions each
- FACC: 40,000 spoken audio versions of those captions (WAV, 16kHz)

Dataset structure expected:
    flickr8k/
        Flicker8k_Dataset/        # Images (note: Flicker not Flickr)
        Flickr8k.token.txt        # Image ID + caption text
        Flickr_8k.trainImages.txt
        Flickr_8k.devImages.txt
        Flickr_8k.testImages.txt
        flickr_audio/             # FACC audio files (optional)
            wav2capt.txt          # Audio file -> (image_id, caption_idx) mapping
            wavs/                 # All .wav files

References:
- Flickr8k: https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html
- FACC: https://sls.csail.mit.edu/downloads/flickraudio/
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Flickr8kAudioDataset(Dataset):
    """Flickr8k + FACC tri-modal dataset.
    
    Args:
        root_dir: Root directory containing flickr8k data
        split: 'train', 'val', or 'test'
        image_size: Target image size (square)
        audio_sample_rate: Target audio sample rate (16kHz default)
        n_mels: Number of mel filterbanks
        text_max_len: Maximum text sequence length
        transform: Optional image transform (overrides default)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        audio_sample_rate: int = 16000,
        n_mels: int = 80,
        text_max_len: int = 77,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.n_mels = n_mels
        self.text_max_len = text_max_len
        
        # Paths
        self.image_dir = self.root_dir / "Flicker8k_Dataset"  # Note: Flicker not Flickr
        self.text_dir = self.root_dir  # Text files are in root, not subdirectory
        self.audio_dir = self.root_dir / "flickr_audio" / "wavs"
        
        # Image transform
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_fft=400,  # 25ms window at 16kHz
            hop_length=160,  # 10ms hop
            n_mels=n_mels,
            f_min=0,
            f_max=8000,
        )
        
        # Load data mappings
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} tri-modal samples for {split} split")
    
    def _load_dataset(self) -> List[Dict]:
        """Load tri-modal (image, text, audio) samples."""
        # Load split image list
        split_map = {
            'train': 'Flickr_8k.trainImages.txt',
            'val': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt',
        }
        
        split_file = self.text_dir / split_map[self.split]
        with open(split_file, 'r') as f:
            split_images = set(line.strip() for line in f)
        
        # Load text captions: image_id -> List[caption]
        captions_file = self.text_dir / "Flickr8k.token.txt"
        image_captions = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Format: "image_id#caption_idx\tcaption_text"
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                img_cap_id, caption = parts
                # Extract image_id and caption_idx
                match = re.match(r'(.+)#(\d+)', img_cap_id)
                if match:
                    img_id, cap_idx = match.groups()
                    cap_idx = int(cap_idx)
                    
                    if img_id not in image_captions:
                        image_captions[img_id] = {}
                    image_captions[img_id][cap_idx] = caption
        
        # Load audio-to-caption mapping
        wav2capt_file = self.root_dir / "flickr_audio" / "wav2capt.txt"
        
        samples = []
        
        if wav2capt_file.exists():
            # Full FACC dataset available
            with open(wav2capt_file, 'r') as f:
                for line in f:
                    # Format varies, but typically: "audio_file.wav image_id caption_idx"
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    audio_file = parts[0]
                    img_id = parts[1]
                    cap_idx = int(parts[2])
                    
                    # Check if image is in this split
                    if img_id in split_images:
                        # Check if caption exists
                        if img_id in image_captions and cap_idx in image_captions[img_id]:
                            samples.append({
                                'image_id': img_id,
                                'caption_idx': cap_idx,
                                'caption': image_captions[img_id][cap_idx],
                                'audio_file': audio_file,
                            })
        else:
            # Fallback: create samples without audio (for testing data loading)
            print("Warning: wav2capt.txt not found. Creating image-text pairs only.")
            for img_id in split_images:
                if img_id in image_captions:
                    for cap_idx, caption in image_captions[img_id].items():
                        samples.append({
                            'image_id': img_id,
                            'caption_idx': cap_idx,
                            'caption': caption,
                            'audio_file': None,
                        })
        
        return samples
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        """Load and transform image with fallback for Flicker vs Flickr naming."""
        image_path = self.image_dir / image_id

        # Try the configured path first
        if not image_path.exists():
            # Fallback: try alternative spellings
            candidates = [
                self.root_dir / "Flickr8k_Dataset" / image_id,
                self.root_dir / "Flicker8k_Dataset" / image_id,
                self.root_dir / "images" / image_id,
                self.root_dir / image_id,
            ]
            for candidate in candidates:
                if candidate.exists():
                    image_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Image not found: {image_id}\n"
                    f"Tried paths:\n  - {self.image_dir / image_id}\n"
                    f"  - {self.root_dir / 'Flickr8k_Dataset' / image_id}\n"
                    f"  - {self.root_dir / 'Flicker8k_Dataset' / image_id}\n"
                    f"Root dir: {self.root_dir}"
                )

        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def _load_audio(self, audio_file: str) -> torch.Tensor:
        """Load and transform audio to mel spectrogram."""
        audio_path = self.audio_dir / audio_file
        
        if not audio_path.exists():
            # Return zeros if audio file missing
            return torch.zeros(self.n_mels, 100)  # Dummy spectrogram
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
        mel_spec = mel_spec.squeeze(0)  # (n_mels, time)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Per-utterance mean/var normalization
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        return mel_spec
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization (replace with your tokenizer).
        
        For real training, use SentencePiece/BPE tokenizer.
        """
        # Lowercase and basic cleanup
        text = text.lower().strip()
        
        # Simple char vocab (a-z, space, punctuation)
        vocab = ' abcdefghijklmnopqrstuvwxyz.,!?\'-'
        char_to_idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 reserved for padding
        
        # Convert to indices
        indices = [char_to_idx.get(c, 0) for c in text[:self.text_max_len]]
        
        # Pad to max length
        if len(indices) < self.text_max_len:
            indices += [0] * (self.text_max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tri-modal sample.
        
        Returns:
            Dictionary with:
                - image: (3, H, W) tensor
                - text: (max_len,) token indices
                - audio: (n_mels, time) mel spectrogram
                - caption_str: original caption string (for evaluation)
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_id'])
        
        # Load text
        text = self._tokenize_text(sample['caption'])
        
        # Load audio (if available)
        if sample['audio_file'] is not None:
            audio = self._load_audio(sample['audio_file'])
        else:
            # Dummy audio if not available
            audio = torch.zeros(self.n_mels, 100)
        
        return {
            'image': image,
            'text': text,
            'audio': audio,
            'caption_str': sample['caption'],
            'image_id': sample['image_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function that pads audio to same length in batch.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary with:
            - images: (B, 3, H, W)
            - text: (B, max_len)
            - audio: (B, n_mels, max_time)
            - caption_strs: List of strings
            - image_ids: List of strings
    """
    images = torch.stack([item['image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    
    # Pad audio to max length in batch
    audios = [item['audio'] for item in batch]
    max_time = max(a.shape[1] for a in audios)
    
    padded_audios = []
    for audio in audios:
        if audio.shape[1] < max_time:
            padding = torch.zeros(audio.shape[0], max_time - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_audios.append(audio)
    
    audios = torch.stack(padded_audios)
    
    caption_strs = [item['caption_str'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'text': texts,
        'audio': audios.unsqueeze(1),  # Add channel dim: (B, 1, n_mels, time)
        'caption_strs': caption_strs,
        'image_ids': image_ids,
    }


# Example usage and dataset download instructions
if __name__ == "__main__":
    print("""
    Flickr8k + FACC Dataset Setup Instructions:
    
    1. Download Flickr8k dataset:
       - Images and captions from: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
       - Text annotations: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
    
    2. Download Flickr Audio Caption Corpus (FACC):
       - From MIT: https://sls.csail.mit.edu/downloads/flickraudio/
       - Or Kaggle mirror: https://www.kaggle.com/datasets/warcoder/flickr-8k-audio-caption-corpus
    
    3. Expected directory structure:
       flickr8k/
           Flickr8k_Dataset/
               *.jpg
           Flickr8k_text/
               Flickr8k.token.txt
               Flickr_8k.trainImages.txt
               Flickr_8k.devImages.txt
               Flickr_8k.testImages.txt
           flickr_audio/
               wav2capt.txt
               wavs/
                   *.wav
    
    4. Usage:
       from flickr8k_dataset import Flickr8kAudioDataset, collate_fn
       from torch.utils.data import DataLoader
       
       dataset = Flickr8kAudioDataset(root_dir='./flickr8k', split='train')
       loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
       
       for batch in loader:
           images = batch['images']  # (B, 3, 224, 224)
           text = batch['text']      # (B, max_len)
           audio = batch['audio']    # (B, 1, n_mels, time)
    """)

