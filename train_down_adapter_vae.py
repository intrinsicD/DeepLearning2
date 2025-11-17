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
from multimodal_brain_v2 import ThinkControl, DownAdapter, MultimodalBrain, ModalityInterface, UpAdapter, ThinkingCore


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


def build_smoke_brain(d_shared: int, device: torch.device):
    """Construct a tiny MultimodalBrain with dummy encoders for smoke testing.
    Returns a model on the requested device.
    """
    # Dummy encoders that return small feature tensors
    class DummyTextEnc(nn.Module):
        def __init__(self, out_dim=64):
            super().__init__()
            self.out_dim = out_dim
        def forward(self, x):
            B = 1
            if isinstance(x, dict):
                if 'input_ids' in x:
                    B = x['input_ids'].size(0)
                else:
                    B = 1
            return torch.zeros(B, 16, self.out_dim)

    class DummyImageEnc(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.out_dim = out_dim
        def forward(self, x):
            # x expected to be dict with 'pixel_values' (B,C,H,W)
            B = x['pixel_values'].size(0)
            T = 8
            return torch.zeros(B, T, self.out_dim)

    class DummyAudioEnc(nn.Module):
        def __init__(self, out_dim=64):
            super().__init__()
            self.out_dim = out_dim
        def forward(self, x):
            B = x['input_features'].size(0) if isinstance(x, dict) and 'input_features' in x else 1
            return torch.zeros(B, 8, self.out_dim)

    text_enc = DummyTextEnc(out_dim=384)
    img_enc = DummyImageEnc(out_dim=768)
    aud_enc = DummyAudioEnc(out_dim=384)

    text_up = UpAdapter(d_in=384, d_shared=d_shared)
    img_up = UpAdapter(d_in=768, d_shared=d_shared)
    aud_up = UpAdapter(d_in=384, d_shared=d_shared)

    # small toy decoder/down below will be attached by caller
    text_iface = ModalityInterface(name='text', encoder=text_enc, up_adapter=text_up, decoder=None, down_adapter=None, freeze_encoder=True, freeze_decoder=True)
    image_iface = ModalityInterface(name='image', encoder=img_enc, up_adapter=img_up, decoder=None, down_adapter=None, freeze_encoder=True, freeze_decoder=True)
    audio_iface = ModalityInterface(name='audio', encoder=aud_enc, up_adapter=aud_up, decoder=None, down_adapter=None, freeze_encoder=True, freeze_decoder=True)

    core = ThinkingCore(d_shared=d_shared, n_layers=2, n_heads=4, dropout=0.0, use_memory_token=False)
    brain = MultimodalBrain(d_shared=d_shared, modalities={'text': text_iface, 'image': image_iface, 'audio': audio_iface}, thinking_core=core, use_memory=False)
    return brain.to(device)


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Device:', device)

    pre = Preproc()

    if args.smoke:
        model = build_smoke_brain(d_shared=args.d_shared, device=device)
    else:
        model = build_brain(d_shared=args.d_shared, device=device, freeze_text=True, freeze_image=True, train_audio_encoder=False)
    model.train()

    # Smoke-mode: replace heavy VAE decoder with a tiny learnable decoder and ensure down-adapter outputs match
    if args.smoke:
        print("Smoke mode: attaching tiny image decoder/down-adapter for fast tests (no HF downloads)")
        # create small down adapter to produce 4*8*8 latents (toy)
        toy_c, toy_h, toy_w = 4, 8, 8
        toy_dout = toy_c * toy_h * toy_w
        # Replace down adapter with a small MLP
        model.modalities['image'].down = DownAdapter(d_shared=args.d_shared, d_out=toy_dout)
        # small decoder: map flattened latents -> 3x64x64 image
        class ToyImageDecoder(nn.Module):
            def __init__(self, d_in: int = toy_dout, out_h: int = 64, out_w: int = 64):
                super().__init__()
                self.out_h = out_h
                self.out_w = out_w
                self.head = nn.Sequential(
                    nn.Linear(d_in, 512),
                    nn.GELU(),
                    nn.Linear(512, 3 * out_h * out_w),
                )
            def forward(self, x: torch.Tensor):
                B = x.size(0)
                if x.dim() == 2:
                    v = x
                else:
                    v = x.view(B, -1)
                out = self.head(v).view(B, 3, self.out_h, self.out_w)
                out = out.sigmoid()
                return out
        model.modalities['image'].decoder = ToyImageDecoder()
        # Ensure decoder parameters require grad (we'll train only down adapter per plan)
        for p in model.modalities['image'].decoder.parameters():
            p.requires_grad = False
        # ensure down-adapter params are trainable
        for p in model.modalities['image'].down.parameters():
            p.requires_grad = True

    # Locate image down adapter
    if 'image' not in model.modalities:
        raise RuntimeError('Model has no image modality')
    img_iface = model.modalities['image']
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
                ctrl = ThinkControl(steps=args.steps)
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
    p.add_argument('--smoke', action='store_true', help='Use tiny toy decoder/down-adapter for quick smoke tests (no HF downloads)')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    train(args)
