"""Improved Flickr8k dataset with proper BPE tokenizer.

Uses a simple BPE tokenizer with 8K vocabulary instead of character-level (33 tokens).
This should provide 20-30% R@1 improvement immediately.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import re

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class Flickr8kImprovedDataset(Dataset):
    """Flickr8k with proper BPE tokenizer (8K vocab).
    
    Args:
        root_dir: Root directory containing flickr8k data
        split: 'train', 'val', or 'test'
        image_size: Target image size
        vocab_size: BPE vocabulary size (default 8192)
        max_length: Maximum sequence length
        tokenizer_path: Path to save/load tokenizer
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        vocab_size: int = 8192,
        max_length: int = 77,
        tokenizer_path: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Paths
        self.image_dir = self.root_dir / "Flicker8k_Dataset"
        
        # Image transform
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        
        # Setup tokenizer
        if tokenizer_path is None:
            tokenizer_path = self.root_dir / f"bpe_tokenizer_{vocab_size}.json"
        self.tokenizer_path = Path(tokenizer_path)
        
        # Load or train tokenizer
        self.tokenizer = self._setup_tokenizer()
        
        # Load data
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")
    
    def _setup_tokenizer(self):
        """Setup BPE tokenizer - load if exists, otherwise train."""
        if self.tokenizer_path.exists():
            print(f"Loading existing tokenizer from {self.tokenizer_path}")
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        else:
            print(f"Training new BPE tokenizer (vocab={self.vocab_size})...")
            
            # Collect all captions for training
            captions_file = self.root_dir / "Flickr8k.token.txt"
            captions = []
            
            with open(captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        captions.append(parts[1].lower())
            
            # Initialize tokenizer
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            
            # Train
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                min_frequency=2,
            )
            
            tokenizer.train_from_iterator(captions, trainer)
            
            # Enable padding and truncation
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.max_length)
            tokenizer.enable_truncation(max_length=self.max_length)
            
            # Save
            tokenizer.save(str(self.tokenizer_path))
            print(f"Tokenizer saved to {self.tokenizer_path}")
        
        return tokenizer
    
    def _load_dataset(self):
        """Load image-text samples."""
        split_map = {
            'train': 'Flickr_8k.trainImages.txt',
            'val': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt',
        }
        
        split_file = self.root_dir / split_map[self.split]
        with open(split_file, 'r') as f:
            split_images = set(line.strip() for line in f)
        
        # Load captions
        captions_file = self.root_dir / "Flickr8k.token.txt"
        image_captions = {}
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                img_cap_id, caption = parts
                match = re.match(r'(.+)#(\d+)', img_cap_id)
                if match:
                    img_id, cap_idx = match.groups()
                    cap_idx = int(cap_idx)
                    
                    if img_id not in image_captions:
                        image_captions[img_id] = {}
                    image_captions[img_id][cap_idx] = caption
        
        # Create samples
        samples = []
        for img_id in split_images:
            if img_id in image_captions:
                for cap_idx, caption in image_captions[img_id].items():
                    samples.append({
                        'image_id': img_id,
                        'caption_idx': cap_idx,
                        'caption': caption,
                    })
        
        return samples
    
    def _load_image(self, image_id: str) -> torch.Tensor:
        """Load and transform image."""
        image_path = self.image_dir / image_id
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using BPE tokenizer."""
        # Tokenize
        encoding = self.tokenizer.encode(text.lower())
        
        # Convert to tensor
        ids = encoding.ids
        
        # Pad or truncate to max_length
        if len(ids) < self.max_length:
            ids = ids + [0] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an image-text sample."""
        sample = self.samples[idx]
        
        image = self._load_image(sample['image_id'])
        text = self._tokenize_text(sample['caption'])
        
        return {
            'image': image,
            'text': text,
            'caption_str': sample['caption'],
            'image_id': sample['image_id'],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = torch.stack([item['image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    
    caption_strs = [item['caption_str'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'text': texts,
        'caption_strs': caption_strs,
        'image_ids': image_ids,
    }

