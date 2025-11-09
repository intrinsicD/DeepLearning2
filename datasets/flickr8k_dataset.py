"""Dataset wrapper exposing train_samples and eval_samples for Flickr8k."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

from torch.utils.data import Dataset

from datasets.raw_flickr8k_dataset import (
    Flickr8kAudioDataset,
    collate_fn as flickr8k_collate_fn,
)


@dataclass
class Flickr8kData:
    root_dir: str = './flickr8k'
    image_size: int = 224
    audio_sample_rate: int = 16000
    n_mels: int = 80
    text_max_len: int = 77

    def train_samples(self) -> Dataset:
        return Flickr8kAudioDataset(
            root_dir=self.root_dir,
            split='train',
            image_size=self.image_size,
            audio_sample_rate=self.audio_sample_rate,
            n_mels=self.n_mels,
            text_max_len=self.text_max_len,
        )

    def eval_samples(self) -> Dataset:
        return Flickr8kAudioDataset(
            root_dir=self.root_dir,
            split='val',
            image_size=self.image_size,
            audio_sample_rate=self.audio_sample_rate,
            n_mels=self.n_mels,
            text_max_len=self.text_max_len,
        )

    @property
    def collate_fn(self) -> Callable:
        return flickr8k_collate_fn


# Re-export a module-level collate function for convenience
collate_fn: Callable = flickr8k_collate_fn

__all__ = ['Flickr8kData', 'collate_fn']
