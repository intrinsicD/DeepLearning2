#!/bin/bash
# Quick TensorBoard Help

cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║              WHY IS TENSORBOARD EMPTY?                         ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║  #1 MOST COMMON REASON (95% of cases):                        ║
║                                                                ║
║  ⏳ TRAINING HASN'T COMPLETED FIRST EPOCH YET                  ║
║                                                                ║
║  ✅ SOLUTION: Just wait! Then refresh browser.                ║
╚════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────┐
│ QUICK CHECK:                                                   │
└────────────────────────────────────────────────────────────────┘

EOF

./check_tensorboard.sh 2>/dev/null | tail -20

cat << 'EOF'

┌────────────────────────────────────────────────────────────────┐
│ WHAT TO EXPECT:                                                │
└────────────────────────────────────────────────────────────────┘

  📊 After Epoch 1 (~5-10 min):
     ✅ Training loss appears in TensorBoard
     ✅ Look for "Loss/train_total" in SCALARS tab

  📊 After Epoch 5 (default --eval_every=5):
     ✅ Validation metrics appear
     ✅ Retrieval R@1 scores show up

┌────────────────────────────────────────────────────────────────┐
│ QUICK ACTIONS:                                                 │
└────────────────────────────────────────────────────────────────┘

  [1] Check detailed status:
      ./check_tensorboard.sh

  [2] Read why it's empty:
      cat WHY_TENSORBOARD_EMPTY.md

  [3] Full troubleshooting:
      cat TENSORBOARD_TROUBLESHOOTING.md

  [4] Restart TensorBoard:
      pkill tensorboard
      tensorboard --logdir=results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard

  [5] Check training terminal for:
      "Epoch 1: 100%" ← When this appears, refresh browser!

┌────────────────────────────────────────────────────────────────┐
│ VERIFICATION:                                                  │
└────────────────────────────────────────────────────────────────┘

  ✅ Training running?
EOF

if pgrep -f train_nlmm > /dev/null; then
    echo "     YES - Training is running"
else
    echo "     NO - Start with: ./train_nlmm.sh --config modules/nl_mm/configs/nano_8gb.yaml"
fi

cat << 'EOF'

  ✅ TensorBoard running?
EOF

if pgrep -f tensorboard > /dev/null; then
    echo "     YES - TensorBoard is running at http://localhost:6006"
else
    echo "     NO - Start with: tensorboard --logdir=results/.../tensorboard"
fi

cat << 'EOF'

  ✅ Event files exist?
EOF

if [ -d "results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard" ]; then
    EVENT_COUNT=$(find results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard -name "*tfevents*" 2>/dev/null | wc -l)
    if [ "$EVENT_COUNT" -gt 0 ]; then
        echo "     YES - Found $EVENT_COUNT event files"
    else
        echo "     NO - No event files yet (training just started)"
    fi
else
    echo "     NO - Directory doesn't exist yet"
fi

cat << 'EOF'

  ✅ Epoch completed?
EOF

if [ -f "results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/metrics.json" ]; then
    echo "     YES - At least one evaluation completed"
else
    echo "     NOT YET - Wait for first epoch to finish"
fi

cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║  REMEMBER:                                                     ║
║                                                                ║
║  📁 Event files created = TensorBoard initialized ✅           ║
║  📊 Data logged = Epoch completed ⏳                           ║
║                                                                ║
║  If training shows "Epoch 1: 100%" but TensorBoard is empty:  ║
║  → Just refresh your browser (Ctrl+R)                         ║
╚════════════════════════════════════════════════════════════════╝

EOF

