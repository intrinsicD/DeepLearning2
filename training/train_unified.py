"""Unified training entry point using NeuralNet wrapper."""
from __future__ import annotations
import argparse

from architectures.factory import build_model
from models.neural_net import NeuralNet, TrainConfig
from datasets.flickr8k_dataset import Flickr8kData
from torch.utils.data import Subset
import torch
from utils.metrics import text_loss_metric


def main():
    parser = argparse.ArgumentParser(description='Unified NL-MM Training')
    parser.add_argument('--model', type=str, default='nlmm')
    parser.add_argument('--config', type=str, default='nl_mm/configs/nano_8gb.yaml')
    parser.add_argument('--data_dir', type=str, default='./flickr8k')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--subset', type=int, default=0, help='Use only N samples for quick runs (0=full)')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer: adamw|adam|sgd|rmsprop|muon|dmgd')
    args = parser.parse_args()

    data = Flickr8kData(root_dir=args.data_dir)
    train_ds = data.train_samples()
    val_ds = data.eval_samples()
    if args.subset and args.subset > 0:
        g = torch.Generator().manual_seed(1337)
        idx_train = torch.randperm(len(train_ds), generator=g)[:args.subset].tolist()
        idx_val = torch.randperm(len(val_ds), generator=g)[:min(args.subset//5+100, len(val_ds))].tolist()
        train_ds = Subset(train_ds, idx_train)
        val_ds = Subset(val_ds, idx_val)

    model = build_model(args.model, config=args.config)
    cfg = TrainConfig(batch_size=args.batch_size, max_epochs=args.epochs, lr=args.lr,
                      amp=args.amp, accumulation_steps=args.accum,
                      log_dir=args.log_dir, checkpoint_dir=args.ckpt_dir,
                      optimizer=args.optimizer)
    net = NeuralNet(model, cfg)

    print('Starting unified training...')
    net.train(train_ds, val_dataset=val_ds, metric_fn=text_loss_metric)
    print('Training complete. Evaluating...')
    eval_res = net.evaluate(val_ds, metric_fn=text_loss_metric)
    print(f"Eval metric: {eval_res['eval_metric']:.4f}")

if __name__ == '__main__':
    main()
