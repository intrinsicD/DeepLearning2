"""Simple Flickr8k Image-Text dataset loader (no audio).

For quick training without FACC audio corpus.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import re

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class Flickr8kImageTextDataset(Dataset):
    """Flickr8k Image-Text dataset (no audio).
    
    Args:
        root_dir: Root directory containing flickr8k data
        split: 'train', 'val', or 'test'
        image_size: Target image size (square)
        text_max_len: Maximum text sequence length
        transform: Optional image transform
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        text_max_len: int = 77,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.text_max_len = text_max_len
        
        # Paths - note the directory is "Flicker8k_Dataset" (with 'e')
        self.image_dir = self.root_dir / "Flicker8k_Dataset"
        
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
        
        # Load data mappings
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} image-text pairs for {split} split")
    
    def _load_dataset(self):
        """Load image-text samples."""
        # Load split image list
        split_map = {
            'train': 'Flickr_8k.trainImages.txt',
            'val': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt',
        }
        
        split_file = self.root_dir / split_map[self.split]
        with open(split_file, 'r') as f:
            split_images = set(line.strip() for line in f)
        
        # Load text captions
        captions_file = self.root_dir / "Flickr8k.token.txt"
        image_captions = {}
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
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
            # Return a blank image on error
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization.
        
        For production, use SentencePiece/BPE tokenizer.
        """
        text = text.lower().strip()
        
        # Simple char vocab
        vocab = ' abcdefghijklmnopqrstuvwxyz.,!?\'-'
        char_to_idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 for padding
        
        # Convert to indices
        indices = [char_to_idx.get(c, 0) for c in text[:self.text_max_len]]
        
        # Pad
        if len(indices) < self.text_max_len:
            indices += [0] * (self.text_max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an image-text sample."""
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_id'])
        
        # Load text
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

