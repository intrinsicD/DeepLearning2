#!/bin/bash
# TensorBoard Status Checker

OUTPUT_DIR="results/folder_per_model/nl_mm/outputs/nlmm_flickr8k"

echo "════════════════════════════════════════════════════════════════"
echo "             TENSORBOARD STATUS DIAGNOSTIC"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Output directory doesn't exist yet"
    echo "   Directory: $OUTPUT_DIR"
    echo ""
    echo "💡 Solution: Start training first!"
    echo "   ./train_nlmm.sh --config modules/nl_mm/configs/nano_8gb.yaml"
    exit 1
fi

echo "✅ Output directory exists: $OUTPUT_DIR"
echo ""

# Check if tensorboard directory exists
if [ ! -d "$OUTPUT_DIR/tensorboard" ]; then
    echo "❌ TensorBoard directory doesn't exist"
    echo ""
    echo "💡 Possible causes:"
    echo "   1. Training hasn't started yet"
    echo "   2. Training used --no_tensorboard flag"
    echo ""
    echo "💡 Solution: Start/restart training without --no_tensorboard"
    exit 1
fi

echo "✅ TensorBoard directory exists"
echo ""

# Check for event files
EVENT_COUNT=$(find "$OUTPUT_DIR/tensorboard" -name "*tfevents*" 2>/dev/null | wc -l)
if [ "$EVENT_COUNT" -eq 0 ]; then
    echo "❌ No TensorBoard event files found"
    echo ""
    echo "💡 This means training hasn't logged anything yet"
    echo "   Wait for at least 1 epoch to complete"
    exit 1
fi

echo "✅ Found $EVENT_COUNT TensorBoard event file(s)"
echo ""

# Check if metrics.json exists
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo "✅ metrics.json exists"

    # Show first few lines
    echo ""
    echo "📊 Metrics preview:"
    head -20 "$OUTPUT_DIR/metrics.json" | sed 's/^/   /'
    echo ""
else
    echo "⚠️  No metrics.json yet"
    echo "   This means evaluation hasn't run yet"
    echo "   (Evaluation runs every --eval_every epochs, default: 5)"
    echo ""
fi

# Check if TensorBoard is running
TB_PIDS=$(pgrep -f "tensorboard.*$OUTPUT_DIR")
if [ -n "$TB_PIDS" ]; then
    echo "✅ TensorBoard is running (PID: $TB_PIDS)"
    echo "   Should be accessible at: http://localhost:6006"
    echo ""
else
    echo "⚠️  TensorBoard is NOT running"
    echo ""
    echo "💡 Start TensorBoard with:"
    echo "   tensorboard --logdir=$OUTPUT_DIR/tensorboard"
    echo ""
fi

# Check if training is running
TRAIN_PIDS=$(pgrep -f "train_nlmm")
if [ -n "$TRAIN_PIDS" ]; then
    echo "✅ Training is running (PID: $TRAIN_PIDS)"
    echo ""
else
    echo "⚠️  Training is NOT running"
    echo ""
    echo "💡 Start training with:"
    echo "   ./train_nlmm.sh --config modules/nl_mm/configs/nano_8gb.yaml"
    echo ""
fi

# Latest event file timestamp
LATEST_EVENT=$(ls -t "$OUTPUT_DIR/tensorboard/"*tfevents* 2>/dev/null | head -1)
if [ -n "$LATEST_EVENT" ]; then
    TIMESTAMP=$(stat -c %y "$LATEST_EVENT" 2>/dev/null || stat -f "%Sm" "$LATEST_EVENT" 2>/dev/null)
    echo "📅 Latest TensorBoard event: $TIMESTAMP"
    echo "   File: $(basename $LATEST_EVENT)"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo "                        SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ "$EVENT_COUNT" -gt 0 ] && [ -n "$TB_PIDS" ]; then
    echo "✅ Everything looks good!"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Open browser to: http://localhost:6006"
    echo "   2. Wait for training to complete at least 1 epoch"
    echo "   3. Refresh browser to see data (Ctrl+R)"
    echo ""
    echo "💡 If TensorBoard is empty:"
    echo "   - Wait for 'Epoch 1: 100%' to complete"
    echo "   - Training loss appears after first epoch"
    echo "   - Validation metrics appear every --eval_every epochs"
    echo ""
elif [ "$EVENT_COUNT" -gt 0 ] && [ -z "$TB_PIDS" ]; then
    echo "⚠️  TensorBoard not running but data exists"
    echo ""
    echo "🎯 Start TensorBoard:"
    echo "   tensorboard --logdir=$OUTPUT_DIR/tensorboard"
    echo ""
elif [ "$EVENT_COUNT" -eq 0 ] && [ -n "$TRAIN_PIDS" ]; then
    echo "⏳ Training is running but no data yet"
    echo ""
    echo "🎯 Wait for first epoch to complete"
    echo "   Then data will appear in TensorBoard"
    echo ""
else
    echo "⚠️  No data and no processes running"
    echo ""
    echo "🎯 Start training:"
    echo "   ./train_nlmm.sh --config modules/nl_mm/configs/nano_8gb.yaml"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📚 For detailed troubleshooting:"
echo "   cat TENSORBOARD_TROUBLESHOOTING.md"
echo ""

