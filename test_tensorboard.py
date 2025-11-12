#!/usr/bin/env python3
"""Quick test to verify TensorBoard logging works"""

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Create test directory
test_dir = Path("test_tensorboard_logging")
test_dir.mkdir(exist_ok=True)

print("Testing TensorBoard logging...")
print(f"Writing to: {test_dir}")

# Create writer
writer = SummaryWriter(str(test_dir))

# Write some test data
for i in range(5):
    writer.add_scalar('Test/loss', 10.0 - i, i)
    writer.add_scalar('Test/accuracy', i * 20, i)
    
writer.flush()
writer.close()

print("✅ Test data written!")
print(f"\nTo verify, run:")
print(f"  tensorboard --logdir={test_dir}")
print(f"  Open http://localhost:6006")
print(f"\nYou should see 'Test/loss' and 'Test/accuracy' in SCALARS tab")

# Verify data was written
import glob
events = glob.glob(str(test_dir / "*tfevents*"))
if events:
    print(f"\n✅ Event file created: {events[0]}")
    
    # Check if it has data
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(events[0])
    ea.Reload()
    scalars = ea.Tags()['scalars']
    
    if scalars:
        print(f"✅ Scalars found: {scalars}")
        for scalar in scalars:
            data = ea.Scalars(scalar)
            print(f"   {scalar}: {len(data)} points")
    else:
        print("❌ No scalar data in event file!")
else:
    print("❌ No event files created!")

