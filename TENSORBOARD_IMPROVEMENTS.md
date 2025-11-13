# TensorBoard Improvements Guide

## What Was Added

Your training script now includes comprehensive TensorBoard logging with better organization and visualization capabilities.

### 1. Hierarchical Organization

Metrics are now organized using `/` separators for better grouping:

```
loss/
  ├── train/
  │   ├── total
  │   ├── text_image
  │   ├── text_audio
  │   ├── image_audio
  │   └── global_modal
  └── val/
      ├── total
      ├── text_image
      ├── text_audio
      ├── image_audio
      └── global_modal

embeddings/
  ├── text/
  │   ├── mean
  │   ├── std
  │   └── norm
  ├── image/
  │   ├── mean
  │   ├── std
  │   └── norm
  ├── audio/
  │   ├── mean
  │   ├── std
  │   └── norm
  └── global/
      ├── mean
      ├── std
      └── norm

optimization/
  ├── lr
  └── grad_norm

alignment/
  ├── text_vs_image/
  │   ├── diagonal_similarity
  │   └── mean_similarity
  ├── text_vs_audio/
  │   ├── diagonal_similarity
  │   └── mean_similarity
  └── image_vs_audio/
      ├── diagonal_similarity
      └── mean_similarity
```

### 2. Custom Dashboard Layouts

The script now sets up custom layouts in TensorBoard:

- **Training Progress**: Shows all loss curves together
- **Embeddings**: Displays embedding statistics (mean, std, norm)
- **Optimization**: Learning rate and gradient norms

### 3. New Visualizations

#### Similarity Matrices
For each epoch, the script generates heatmaps showing cosine similarity between modalities:
- Text vs Image
- Text vs Audio  
- Image vs Audio

These help you visualize how well aligned different modalities are.

#### Weight & Gradient Histograms
Every 5 epochs (configurable), the script logs:
- Weight distributions for all trainable parameters
- Gradient distributions to detect vanishing/exploding gradients

#### Embedding Projections
Text embeddings are projected using t-SNE/UMAP in TensorBoard's PROJECTOR tab, with captions as metadata.

### 4. Comprehensive Metrics

**Per-step logging** (every 10 steps):
- Individual loss components
- Learning rate
- Gradient norms
- Embedding statistics (mean, std, norm) per modality

**Per-epoch logging**:
- Epoch mean and std of training loss
- Validation losses (averaged)
- Sample images and captions
- Similarity matrices
- Alignment metrics

### 5. Configuration Options

You can customize logging intervals:

```python
Config(
    log_interval=10,        # Log scalars every 10 steps
    histogram_interval=5,   # Log histograms every 5 epochs
)
```

Or via command line:
```bash
python train_multimodal_brain_flickr8k.py \
    --root_dir data/flickr8k \
    --epochs 50 \
    --batch_size 16
```

## How to View TensorBoard

### Start TensorBoard

```bash
tensorboard --logdir=runs_brain_v2 --port=6006
```

Then open http://localhost:6006 in your browser.

### Navigate the Interface

1. **SCALARS Tab**: 
   - Use the custom layouts in the dropdown at the top
   - Toggle log scale for better visualization
   - Smooth curves using the smoothing slider

2. **IMAGES Tab**:
   - View sample images from validation set
   - See similarity matrices as heatmaps

3. **GRAPHS Tab**:
   - Visualize model architecture

4. **DISTRIBUTIONS/HISTOGRAMS Tabs**:
   - Monitor weight and gradient distributions
   - Identify potential training issues (vanishing/exploding gradients)

5. **PROJECTOR Tab**:
   - Explore text embeddings in 3D
   - Search for specific captions
   - See clusters of similar captions

6. **TEXT Tab**:
   - Read sample captions from validation set

## What to Look For

### Healthy Training Signs
✅ Loss curves decreasing steadily
✅ Gradient norms stable (not too small, not exploding)
✅ Embedding norms relatively stable
✅ Similarity matrices showing diagonal structure (good alignment)
✅ Diagonal similarity > mean similarity (correct pairs more similar)

### Warning Signs
⚠️ Loss plateauing or increasing
⚠️ Gradient norms approaching 0 (vanishing gradients)
⚠️ Gradient norms exploding (>100)
⚠️ Embedding norms collapsing to 0
⚠️ No diagonal structure in similarity matrices

## Tips

1. **Compare Runs**: TensorBoard can overlay multiple runs. Just keep different run directories under `runs_brain_v2/`.

2. **Refresh**: TensorBoard auto-refreshes, but you can force refresh with the refresh button.

3. **Download Data**: You can download raw data as CSV or JSON from any chart.

4. **Share Results**: Use TensorBoard.dev to share your results online:
   ```bash
   tensorboard dev upload --logdir runs_brain_v2
   ```

5. **Filter Metrics**: Use regex in the filter box to show only specific metrics.

## Troubleshooting

### TensorBoard shows nothing
- Make sure training has started and logged at least one step
- Check that the logdir path is correct
- Try restarting TensorBoard

### Old data still showing
- Clear browser cache or use incognito mode
- Delete old event files in runs_brain_v2/

### Similarity matrices not showing
- Requires matplotlib: `pip install matplotlib`
- Check the IMAGES tab, not SCALARS

### Projector not working
- Needs enough samples (at least 100)
- Only shows text embeddings currently
- Embeddings logged at end of each epoch

## Example Analysis Workflow

1. Start training and TensorBoard
2. Monitor loss curves in real-time (SCALARS tab)
3. Check gradient norms to ensure stable training (SCALARS > Optimization)
4. Every few epochs, check similarity matrices (IMAGES tab)
5. Look at embedding statistics to ensure they're not collapsing
6. After training, explore embedding clusters (PROJECTOR tab)
7. Compare validation samples with their captions (TEXT + IMAGES tabs)

## Advanced: Adding Custom Metrics

To add your own metrics, edit the training script:

```python
# In the train loop
self.writer.add_scalar("my_metric/name", value, self.global_step)

# For images
self.writer.add_image("my_images/name", image_tensor, epoch)

# For text
self.writer.add_text("my_text/name", text_string, epoch)
```

Enjoy your enhanced TensorBoard experience! 🎉

