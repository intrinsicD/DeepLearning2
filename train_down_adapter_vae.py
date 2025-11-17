# train_down_adapter_vae.py
"""Train only the image down-adapter to map shared latents into a pretrained VAE latent space.

Usage examples:
  python train_down_adapter_vae.py --flickr_root flickr8k --epochs 5 --batch_size 16 --lr 1e-3 --device cuda:0

Notes:
- This script assumes `build_brain` in `brain_v2_components.py` attaches a VAE-based image decoder
  (AutoencoderKL) and sets the image down_adapter to output flattened VAE latents (C*H*W).
- The VAE decoder is frozen; we train only the `modalities['image'].down` parameters.
"""
import argparse
from pathlib import Path
import os
from PIL import Image
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from brain_v2_components import Preproc, build_brain


class FlickrImageFilesDataset(Dataset):
    def __init__(self, root: str, split: str = 'train'):
        root = Path(root)
        split_map = {
            'train': 'Flickr_8k.trainImages.txt',
            'val': 'Flickr_8k.devImages.txt',
            'test': 'Flickr_8k.testImages.txt'
        }
        split_file = root / split_map.get(split, '')
        img_dir = root / 'Flicker8k_Dataset'
        if split_file.exists():
            names = [ln.strip() for ln in open(split_file, 'r', encoding='utf-8') if ln.strip()]
            paths = [img_dir / n for n in names]
            self.paths = [str(p) for p in paths if p.exists()]
        else:
            self.paths = [str(p) for p in img_dir.glob('*.jpg')]
        self.target_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert('RGB')
        target = self.target_transform(img)  # 3x256x256 in [0,1]
        return {'pil': img, 'target': target, 'path': p}


def collate_fn(batch: List[dict]):
    targets = torch.stack([b['target'] for b in batch], dim=0)
    pil = [b['pil'] for b in batch]
    paths = [b['path'] for b in batch]
    return {'pil': pil, 'target': targets, 'path': paths}


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Device:', device)

    pre = Preproc()

    model = build_brain(d_shared=args.d_shared, device=device, freeze_text=True, freeze_image=True, train_audio_encoder=False)
    model.train()

    # Locate image down adapter
    img_iface = model.modalities.get('image')
    if img_iface is None:
        raise RuntimeError('Model has no image modality')
    if img_iface.down is None:
        raise RuntimeError('Image modality has no down adapter to train')

    # Train only the down-adapter parameters
    params = [p for p in img_iface.down.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('No trainable parameters found in image down adapter')

    opt = optim.Adam(params, lr=args.lr, weight_decay=1e-6)
    loss_fn = nn.L1Loss()

    ds = FlickrImageFilesDataset(args.flickr_root, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        running = 0.0
        n = 0
        for batch in loader:
            pil_imgs = batch['pil']
            targets = batch['target'].to(device)

            # Prepare CLIP inputs for encoder via Preproc.clip_proc
            px = pre.clip_proc(images=pil_imgs, return_tensors='pt')
            pixel_values = px.pixel_values.to(device)
            inputs = {'image': {'pixel_values': pixel_values}}

            # Forward through model: encode -> think -> decode image via VAE decoder
            opt.zero_grad()
            with torch.autocast(device.type, enabled=(args.precision=='fp16' and device.type=='cuda')):
                z_by_mod = model.encode_inputs(inputs)
                ctrl = type('C', (), {'steps': args.steps, 'mode':'default', 'effective_steps': lambda self: args.steps})()
                tokens, z_global, z_by_mod_out = model.think(z_by_mod, ctrl)
                decoded = model.decode_outputs(z_global, z_by_mod_out, ['image'])
                img_out = decoded.get('image')
                if img_out is None:
                    print('Warning: model returned no image; skipping')
                    continue
                # img_out expected in [0,1] float shape (B,3,256,256)
                if isinstance(img_out, dict):
                    # find first tensor
                    for v in img_out.values():
                        if torch.is_tensor(v):
                            img_out = v
                            break
                if not torch.is_tensor(img_out):
                    print('Decoded image not tensor, skipping batch')
                    continue
                img_out = img_out.to(device)
                # ensure same spatial size
                if img_out.shape[2:] != targets.shape[2:]:
                    img_out = nn.functional.interpolate(img_out, size=targets.shape[2:], mode='bilinear', align_corners=False)
                loss = loss_fn(img_out, targets)
            loss.backward()
            opt.step()

            running += loss.item() * targets.size(0)
            n += targets.size(0)
        avg = running / max(1, n)
        print(f'[Epoch {epoch}] image L1={avg:.6f}')
        # save adapter checkpoint
        ckpt = {
            'image_down_state': img_iface.down.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.outdir, f'image_down_epoch{epoch}.pt'))

    print('Done')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--flickr_root', type=str, default='flickr8k')
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--d_shared', type=int, default=512)
    p.add_argument('--precision', type=str, default='fp32', choices=['fp32','fp16'])
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--outdir', type=str, default='checkpoints')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    train(args)

