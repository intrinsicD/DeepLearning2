#!/bin/bash
# Fix TensorBoard - Restart training with updated logging code

echo "════════════════════════════════════════════════════════════════"
echo "              TENSORBOARD LOGGING FIX"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🔍 PROBLEM IDENTIFIED:"
echo "   Your current training is using OLD code (started 13:18)"
echo "   The fixed logging code was added AFTER your training started"
echo ""
echo "✅ SOLUTION:"
echo "   Restart training to load the updated code with TensorBoard logging"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if training is running
if pgrep -f train_nlmm_flickr8k.py > /dev/null; then
    echo "⚠️  Training is currently running (PID: $(pgrep -f train_nlmm_flickr8k.py | head -1))"
    echo ""
    echo "📋 OPTIONS:"
    echo ""
    echo "   [1] Stop current training and restart with fixed code"
    echo "       (You'll lose progress from epoch 4)"
    echo ""
    echo "   [2] Let current training finish, then start new run"
    echo "       (Current run won't have TensorBoard data)"
    echo ""
    echo "   [3] Continue current training and manually add logging"
    echo "       (Advanced - requires code changes)"
    echo ""
    read -p "Choose option (1/2/3) or 'q' to quit: " choice

    case $choice in
        1)
            echo ""
            echo "🛑 Stopping current training..."
            pkill -f train_nlmm_flickr8k.py
            sleep 2

            echo "✅ Training stopped"
            echo ""
            echo "🚀 Starting new training with TensorBoard logging..."
            echo ""

            # Backup old tensorboard data
            if [ -d "results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard" ]; then
                mv results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard \
                   results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard_old_$(date +%s)
                echo "📁 Backed up old (empty) TensorBoard logs"
            fi

            echo ""
            echo "Starting training with fixed code..."
            ./train_nlmm.sh \
                --config modules/nl_mm/configs/nano_8gb.yaml \
                --epochs 50 \
                --batch_size 16 \
                --eval_every 1
            ;;
        2)
            echo ""
            echo "✅ Leaving current training running"
            echo ""
            echo "💡 After it finishes, start new training with:"
            echo "   ./train_nlmm.sh --config modules/nl_mm/configs/nano_8gb.yaml"
            echo ""
            ;;
        3)
            echo ""
            echo "⚠️  Advanced option - requires manual intervention"
            echo ""
            echo "The training process has already loaded the old code into memory."
            echo "You cannot update it without restarting."
            echo ""
            echo "Recommendation: Choose option 1 or 2"
            echo ""
            ;;
        *)
            echo ""
            echo "Cancelled"
            ;;
    esac
else
    echo "✅ No training running currently"
    echo ""
    echo "🚀 Start training with TensorBoard logging:"
    echo ""
    echo "   ./train_nlmm.sh \\"
    echo "       --config modules/nl_mm/configs/nano_8gb.yaml \\"
    echo "       --epochs 50 \\"
    echo "       --batch_size 16 \\"
    echo "       --eval_every 1"
    echo ""
    read -p "Start training now? (y/n): " start

    if [ "$start" = "y" ] || [ "$start" = "Y" ]; then
        echo ""
        echo "🚀 Starting training..."
        ./train_nlmm.sh \
            --config modules/nl_mm/configs/nano_8gb.yaml \
            --epochs 50 \
            --batch_size 16 \
            --eval_every 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 After restarting, TensorBoard will show data after epoch 1!"
echo ""
echo "To monitor:"
echo "  tensorboard --logdir=results/folder_per_model/nl_mm/outputs/nlmm_flickr8k/tensorboard"
echo ""

