#!/usr/bin/env python3
"""Diagnose training stability issues - WITH gradient clipping"""

import torch
import torch.nn.functional as F
from train_multimodal_sota8gb import Config, Trainer, ThinkControl, info_nce

def diagnose():
    cfg = Config(demo=True, epochs=1, batch_size=4, grad_clip=1.0)
    trainer = Trainer(cfg)
    
    print("="*80)
    print("TRAINING DIAGNOSTICS (WITH GRADIENT CLIPPING)")
    print("="*80)
    
    # Check parameter stats
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params, {trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)")
    print(f"Gradient clipping: {cfg.grad_clip}")
    print(f"Learning rate: {cfg.lr}")
    
    # Run several training steps
    trainer.model.train()
    losses = []
    grad_norms_clipped = []
    grad_norms_unclipped = []
    
    for i, batch in enumerate(trainer.train_loader):
        if i >= 10:
            break
            
        inputs = trainer._to_device_nested(batch)
        
        with torch.amp.autocast('cuda', enabled=(trainer.precision!="fp32" and trainer.device.type=="cuda")):
            modal_lat, _ = trainer._encode_modal(inputs)
            outs = trainer.model(inputs, request_outputs=[], control=ThinkControl(steps=2))
            Z = outs['_Z']
            Zp = Z.mean(dim=1)
            
            # Compute losses
            align_loss = 0.0
            cnt = 0
            for m, z_m in modal_lat.items():
                zt = trainer.model.modal2think[m](z_m).mean(dim=1)
                align_loss = align_loss + info_nce(Zp, zt, t=cfg.info_nce_temp)
                cnt += 1
            if cnt > 0:
                align_loss = align_loss / cnt
                
            recon_loss = 0.0
            cnt2 = 0
            for m, z_m in modal_lat.items():
                z_from_think = trainer.model.modal2think[m](z_m).mean(dim=1)
                recon_loss = recon_loss + (1.0 - F.cosine_similarity(Zp, z_from_think, dim=-1).mean())
                cnt2 += 1
            if cnt2 > 0:
                recon_loss = recon_loss / cnt2
                
            total = 0.6 * align_loss + 0.4 * recon_loss
        
        # Backward
        trainer.optim.zero_grad(set_to_none=True)
        
        if trainer.scaler.is_enabled():
            trainer.scaler.scale(total).backward()
            trainer.scaler.unscale_(trainer.optim)
        else:
            total.backward()
        
        # Compute UNCLIPPED gradient norm
        total_norm_unclipped = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_unclipped += param_norm.item() ** 2
        total_norm_unclipped = total_norm_unclipped ** 0.5
        grad_norms_unclipped.append(total_norm_unclipped)
        
        # Apply gradient clipping
        grad_norm_clipped = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.cfg.grad_clip)
        grad_norms_clipped.append(grad_norm_clipped.item())
        
        # Step optimizer
        if trainer.scaler.is_enabled():
            trainer.scaler.step(trainer.optim)
            trainer.scaler.update()
        else:
            trainer.optim.step()
        
        trainer.scheduler.step()
        
        losses.append(total.item())
        
        # Check for NaN/Inf
        if torch.isnan(total) or torch.isinf(total):
            print(f"⚠️  Step {i}: NaN or Inf detected in loss!")
            return
    
    # Summary
    print(f"\n{'Step':<6} {'Loss':<12} {'Grad (unclipped)':<18} {'Grad (clipped)':<15}")
    print("-" * 55)
    for i, (loss, gunclip, gclip) in enumerate(zip(losses, grad_norms_unclipped, grad_norms_clipped)):
        reduction = (gunclip - gclip) / gunclip * 100 if gunclip > 0 else 0
        marker = "✂️" if gclip < gunclip else ""
        print(f"{i:<6} {loss:<12.6f} {gunclip:<18.2f} {gclip:<15.4f} {marker}")
    
    print(f"\nLoss: mean={sum(losses)/len(losses):.6f}, min={min(losses):.6f}, max={max(losses):.6f}")
    print(f"Grad (unclipped): mean={sum(grad_norms_unclipped)/len(grad_norms_unclipped):.2f}, max={max(grad_norms_unclipped):.2f}")
    print(f"Grad (clipped): mean={sum(grad_norms_clipped)/len(grad_norms_clipped):.4f}, max={max(grad_norms_clipped):.4f}")
    
    avg_reduction = sum((u - c) / u * 100 for u, c in zip(grad_norms_unclipped, grad_norms_clipped)) / len(grad_norms_unclipped)
    print(f"Average gradient reduction: {avg_reduction:.1f}%")
    
    # Check if loss is decreasing
    if losses[-1] < losses[0] * 0.95:
        print("\n✓ Loss is decreasing (>5% improvement)")
    elif losses[-1] < losses[0]:
        print("\n⚙️  Loss decreasing slightly (<5% improvement)")
    else:
        print("\n⚠️  Warning: Loss not decreasing in first 10 steps")
        print("   This is normal during warmup. Check after 100+ steps.")
    
    # Check gradient norms (after clipping)
    if max(grad_norms_clipped) > cfg.grad_clip * 1.1:
        print(f"❌ Error: Gradient clipping not working properly!")
    elif all(g < cfg.grad_clip * 0.99 for g in grad_norms_clipped):
        print(f"✓ Gradient clipping working (all norms ≈ {cfg.grad_clip})")
    else:
        print(f"✓ Gradient clipping active")

if __name__ == "__main__":
    diagnose()

