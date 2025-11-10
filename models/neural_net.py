"""Unified NeuralNet interface for the repository.

Provides a consistent API:
    net = NeuralNet(model)
    net.train(train_dataset)
    metrics = net.evaluate(val_dataset)
    output = net.forward(batch)

Features:
- NaN / Inf detection early
- Best model checkpointing
- TensorBoard logging
- Graceful anomaly abort
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    # Optional import for known datasets
    from datasets.flickr8k_dataset import collate_fn as flickr8k_collate
except Exception:
    flickr8k_collate = None


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 10
    lr: float = 3e-4
    grad_clip: float = 1.0
    amp: bool = False
    accumulation_steps: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir: str = './runs'
    checkpoint_dir: str = './checkpoints'
    eval_every: int = 1
    save_best: bool = True
    nan_abort: bool = True
    pin_memory: bool = True
    optimizer: str = 'adamw'


class NeuralNet:
    def __init__(self, model: nn.Module, config: Optional[TrainConfig] = None):
        self.model = model
        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.writer: Optional[SummaryWriter] = None
        self.global_step = 0
        self.best_metric: float = -1e9
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.amp)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _init_writer(self):
        if self.writer is None:
            ts = time.strftime('%Y%m%d-%H%M%S')
            run_dir = Path(self.cfg.log_dir) / f"run-{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(run_dir))

    def _adapt_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Map dataset batch keys/shapes to model-expected format.
        - images -> image
        - audio: (B, 1, n_mels, t) -> (B, 1, n_mels*t)
        - add text_target if text exists
        Move tensors to device.
        """
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        if 'images' in out and 'image' not in out:
            out['image'] = out.pop('images')
        if 'audio' in out and out['audio'] is not None and torch.is_tensor(out['audio']):
            # Accept either (B,1,M,T) or (B,1,L). If 4D, convert to 3D length.
            if out['audio'].ndim == 4:
                B, C, M, T = out['audio'].shape
                out['audio'] = out['audio'].reshape(B, C, M * T)
        if 'text' in out and 'text_target' not in out and torch.is_tensor(out['text']):
            out['text_target'] = out['text']
        return out

    def _extract_loss(self, out: Any) -> Optional[torch.Tensor]:
        """Handle different model output conventions: dict or tuple."""
        # NLMM returns (outputs_dict, state)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            return out.get('loss') or out.get('text')
        if torch.is_tensor(out):
            return out
        return None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            batch = self._adapt_batch(batch)
            out = self.model(batch)
            # Normalize to dict
            if isinstance(out, tuple):
                out = out[0]
        return out  # type: ignore[return-value]

    def _check_anomaly(self, tensor: torch.Tensor, name: str) -> bool:
        if tensor is None:
            return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"❌ Anomaly detected in {name}. Aborting training.")
            return True
        return False

    def _save_checkpoint(self, epoch: int, is_best: bool = False, metric: float = 0.0):
        payload = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'config': self.cfg.__dict__,
            'metric': metric,
            'global_step': self.global_step,
        }
        path = Path(self.cfg.checkpoint_dir) / f'epoch{epoch}.pt'
        torch.save(payload, path)
        if is_best:
            torch.save(payload, Path(self.cfg.checkpoint_dir) / 'best.pt')
            print(f"💾 Saved best model (metric={metric:.4f})")

    def evaluate(self, dataset, metric_fn=None) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=flickr8k_collate if flickr8k_collate is not None else None,
        )
        agg = []
        with torch.no_grad():
            for batch in loader:
                batch = self._adapt_batch(batch)
                out = self.model(batch)
                loss = self._extract_loss(out)
                if torch.is_tensor(loss):
                    if self._check_anomaly(loss, 'eval_loss') and self.cfg.nan_abort:
                        break
                if metric_fn:
                    # Ensure out is dict
                    out_dict = out[0] if isinstance(out, tuple) else out
                    agg.append(metric_fn(batch, out_dict))
        result = {'eval_metric': sum(agg)/len(agg) if agg else 0.0}
        if self.writer:
            self.writer.add_scalar('eval/metric', result['eval_metric'], self.global_step)
        return result

    def train(self, dataset, val_dataset=None, metric_fn=None):
        self._init_writer()
        # Optimizer factory
        opt_name = (self.cfg.optimizer or 'adamw').lower()
        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        elif opt_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.cfg.lr)
        else:
            # Attempt custom registry
            optimizer = None
            try:
                from optimizers.universal_optimizers import UniversalMuon
                if opt_name in ('muon', 'universalmuon'):
                    optimizer = UniversalMuon(self.model.parameters(), lr=self.cfg.lr)
            except Exception:
                pass
            try:
                from modules.nl_mm.modules.optim.d_mgd import DMGD
                if opt_name == 'dmgd':
                    optimizer = DMGD(self.model.parameters(), lr=self.cfg.lr, beta=0.9, learnable_modulation=True)
            except Exception:
                pass
            if optimizer is None:
                raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=flickr8k_collate if flickr8k_collate is not None else None,
        )
        # Sanity check params
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"❌ NaN/Inf in parameter {name} before training")
                    return
        for epoch in range(1, self.cfg.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                batch = self._adapt_batch(batch)
                with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                    out = self.model(batch)
                    loss = self._extract_loss(out)
                if loss is None:
                    continue
                if self._check_anomaly(loss, 'train_loss') and self.cfg.nan_abort:
                    return
                if self.cfg.accumulation_steps > 1:
                    loss = loss / self.cfg.accumulation_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    if self.cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.detach().cpu())
                self.global_step += 1
                if self.writer:
                    self.writer.add_scalar('train/loss', float(loss.detach().cpu()), self.global_step)
            avg_loss = epoch_loss / max(1, len(loader))
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
            if val_dataset and (epoch % self.cfg.eval_every == 0):
                eval_res = self.evaluate(val_dataset, metric_fn)
                metric = eval_res['eval_metric']
                improved = metric > self.best_metric
                if improved:
                    self.best_metric = metric
                if self.cfg.save_best:
                    self._save_checkpoint(epoch, is_best=improved, metric=metric)
        if self.writer:
            self.writer.close()

__all__ = ['NeuralNet', 'TrainConfig']
