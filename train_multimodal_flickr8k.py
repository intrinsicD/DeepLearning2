"""
Train the multimodal thinking model on Flickr8k dataset.

Usage:
    python train_multimodal_flickr8k.py --epochs 10 --batch_size 8
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Import the existing Flickr8k dataset
from datasets.flickr8k_dataset import Flickr8kData

# Import the training infrastructure from the main script
# We'll reuse the Config, Trainer, and model building functions
import train_multimodal_sota8gb as mm_train


def parse_args():
    parser = argparse.ArgumentParser(description='Train multimodal model on Flickr8k')
    parser.add_argument('--root_dir', type=str, default='data/flickr8k',
                        help='Path to Flickr8k root directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Training precision')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--savedir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--logdir', type=str, default='runs',
                        help='Directory for tensorboard logs')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if Flickr8k exists
    root_path = Path(args.root_dir)
    if not root_path.exists():
        print(f"Error: Flickr8k directory not found at {args.root_dir}")
        print("Please ensure the dataset is downloaded and the path is correct.")
        return
    
    # Create the configuration
    cfg = mm_train.Config(
        demo=False,  # Not using demo mode
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        precision=args.precision,
        checkpoint=args.checkpoint,
        savedir=args.savedir,
        logdir=args.logdir,
    )
    
    # Initialize the Flickr8k dataset
    print(f"Loading Flickr8k dataset from {args.root_dir}...")
    flickr_data = Flickr8kData(root_dir=args.root_dir)
    
    # Get train and validation datasets
    train_dataset = flickr_data.train_samples()
    val_dataset = flickr_data.eval_samples()
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create custom trainer that uses Flickr8k dataset
    class Flickr8kTrainer(mm_train.Trainer):
        def __init__(self, cfg, train_data, val_data):
            # Call parent init but skip data loading
            self.cfg = cfg
            self.device = mm_train.get_device()
            os.makedirs(cfg.logdir, exist_ok=True)
            os.makedirs(cfg.savedir, exist_ok=True)
            self.writer = mm_train.SummaryWriter(log_dir=os.path.join(cfg.logdir, "MMT_Flickr8k"))
            mm_train.set_seed(1234)
            self.precision = cfg.precision
            
            # Initialize preprocessing BEFORE creating loaders
            self.pre = mm_train.Preproc()

            # Use provided datasets instead of creating them
            self.train_data = train_data
            self.val_data = val_data
            
            # Create data loaders - use built-in collate for now, we'll wrap the dataset
            from datasets.flickr8k_dataset import collate_fn as flickr_collate
            self.train_loader = DataLoader(
                self.train_data, 
                batch_size=cfg.batch_size, 
                shuffle=True, 
                num_workers=0,  # Disable multiprocessing for stability
                collate_fn=flickr_collate
            )
            self.val_loader = DataLoader(
                self.val_data, 
                batch_size=cfg.batch_size, 
                shuffle=False, 
                num_workers=0,  # Disable multiprocessing for stability
                collate_fn=flickr_collate
            )
            
            # Build model
            self.model = mm_train.build_model(cfg, self.device)
            
            # Setup optimizer
            params = [p for p in self.model.parameters() if p.requires_grad]
            if cfg.use_8bit_optim and mm_train.HAS_BNB:
                self.optim = mm_train.bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            
            self.scaler = torch.amp.GradScaler('cuda', enabled=(self.precision=="fp16" and self.device.type=="cuda"))
            
            # Learning rate scheduler with warmup
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
            
            self.best_val = float("inf")
            self.global_step = 0
            self.nan_count = 0
            self.max_nan_tolerance = 5
        
        def _convert_flickr_batch(self, flickr_batch):
            """
            Convert Flickr8k batch format to multimodal model format.

            Flickr8k format: {
                'images': (B, 3, H, W),
                'text': (B, max_len) tokenized indices,
                'audio': (B, 1, n_mels, time),
                'caption_strs': List[str],
                'image_ids': List[str]
            }

            Multimodal format: {
                'text': {'input_ids': ..., 'attention_mask': ...},
                'image': {'pixel_values': ...},
                'audio': {'input_features': ...}
            }
            """
            result = {}
            
            # Convert text
            if 'caption_strs' in flickr_batch:
                texts = flickr_batch['caption_strs']
                tk = self.pre.txt_tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                result["text"] = {"input_ids": tk.input_ids, "attention_mask": tk.attention_mask, "raw": texts}
            
            # Convert images - denormalize and reprocess with CLIP
            if 'images' in flickr_batch:
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                pil_images = []
                for img in flickr_batch['images']:
                    img_denorm = img * std + mean
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    pil_images.append(to_pil(img_denorm))
                px = self.pre.clip_proc(images=pil_images, return_tensors="pt")
                result["image"] = {"pixel_values": px.pixel_values}
            
            # Convert audio - create dummy audio since we only have mel specs
            if 'audio' in flickr_batch:
                import numpy as np
                B = flickr_batch['audio'].shape[0]
                # Create silent 3-second audio for each sample
                dummy_wavs = [np.zeros(48000, dtype=np.float32) for _ in range(B)]
                feats = self.pre.whisper_feat(dummy_wavs, sampling_rate=16000, return_tensors="pt")
                result["audio"] = {"input_features": feats.input_features}
            
            return result

        def _to_device_nested(self, inputs):
            """Override to convert Flickr format first, then move to device."""
            # Convert format
            converted = self._convert_flickr_batch(inputs)
            # Move to device
            out = {}
            for k, v in converted.items():
                out[k] = {kk: (vv.to(self.device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
            return out

    # Create trainer
    print("Initializing trainer...")
    trainer = Flickr8kTrainer(cfg, train_dataset, val_dataset)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.model.load_state_dict(torch.load(args.checkpoint, map_location=trainer.device))
    
    # Train
    print("Starting training...")
    print(f"Device: {trainer.device}")
    print(f"Precision: {cfg.precision}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Checkpoints will be saved to: {cfg.savedir}")
    print(f"TensorBoard logs will be saved to: {cfg.logdir}")
    print("-" * 80)
    
    trainer.train()
    
    print("-" * 80)
    print("Training complete!")
    print(f"Best validation loss: {trainer.best_val:.4f}")
    print(f"Best model saved to: {os.path.join(cfg.savedir, 'best.pt')}")


if __name__ == "__main__":
    main()

