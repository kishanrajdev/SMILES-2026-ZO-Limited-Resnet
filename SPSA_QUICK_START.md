# SPSA Quick Start Guide

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Optimizer

### Basic Run (Default SPSA Configuration)

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 32 \
    --output results.json
```

This runs with:
- **SPSA enabled** by default
- **Momentum**: 0.9 (smooths gradient estimates)
- **Learning rate**: 1e-3
- **Perturbation scale (eps)**: 1e-3
- **Total budget**: 32 batches × 32 samples = 1024 samples
- **Forward passes**: ~64 (2 per SPSA step)

### Larger Budget (More Optimization Steps)

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 16 \
    --n_batches 64 \
    --output results.json
```

Trade-off: 64 steps with noisier (smaller batch) loss estimates vs 32 steps with cleaner estimates.

### Maximum Budget

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 8 \
    --n_batches 128 \
    --output results.json
```

Extreme case: 128 optimization steps with very noisy gradients. Momentum becomes even more important here.

## Customizing the Optimizer

Edit `zo_optimizer.py` to customize:

### 1. Change Learning Rate
```python
optimizer = ZeroOrderOptimizer(model, lr=5e-3)  # Increase from 1e-3
```

### 2. Tune Perturbation Scale
```python
optimizer = ZeroOrderOptimizer(model, eps=5e-3)  # Larger perturbations
```

### 3. Adjust Momentum
```python
optimizer = ZeroOrderOptimizer(model, momentum=0.95)  # More aggressive smoothing
```

### 4. Use Uniform Perturbations Instead of Gaussian
```python
optimizer = ZeroOrderOptimizer(model, perturbation_mode="uniform")
```

### 5. Disable SPSA (Fall Back to Per-Parameter Method)
```python
optimizer = ZeroOrderOptimizer(model, use_spsa=False)
```
⚠️ **Warning**: This uses 2×|active_params| forward passes per step - very expensive!

## Selecting Layers to Optimize

### Current Default: Head Only
```python
optimizer.layer_names = ["fc.weight", "fc.bias"]
```

### Progressive Unfreezing: Head + Layer 4
```python
optimizer.layer_names = [
    "layer4.1.conv2.weight",
    "layer4.1.bn2.weight", 
    "layer4.1.bn2.bias",
    "fc.weight", 
    "fc.bias"
]
```

### Curriculum Strategy: Implement in Model
```python
# In model.py or validate.py, modify optimizer layers partway through:
for step in range(n_batches):
    if step == n_batches // 2:
        # Unfreeze deeper layers halfway through
        optimizer.layer_names = ["layer4.0.conv1.weight", ...] + original_layers
    loss = optimizer.step(loss_fn)
```

## Understanding the Output

`results.json` contains:

```json
{
  "val_accuracy_imagenet_head": 0.04,
  "val_accuracy_init_head": 0.15,
  "val_accuracy_finetuned": 0.35,
  "n_batches": 32,
  "batch_size": 32,
  "layers_tuned": ["fc.weight", "fc.bias"],
  "total_samples": 1024
}
```

**Key metric**: `val_accuracy_finetuned` - this is what gets graded.

### Interpreting Gaps

- **Gap (init → finetuned)**: Shows how much SPSA improved the model
  - Small gap = optimizer struggling or learning rate too low
  - Large gap = optimizer working well
  
- **Baseline (imagenet_head)**: Should be very low (~4%) - sanity check

- **Init head**: Reflects your initialization strategy quality

## Troubleshooting

### Loss Not Decreasing?
1. Increase `lr` (try 5e-3 or 1e-2)
2. Increase `eps` (try 5e-3 or 1e-2)
3. Use larger batch size for less noisy estimates
4. Check `augmentation.py` for better training data

### Loss Oscillating/Unstable?
1. Decrease `lr` (try 5e-4)
2. Decrease `eps` (try 5e-4)
3. Increase `momentum` (try 0.95 or 0.99)
4. Use smaller batch size to prevent overfitting

### Running Out of Memory?
1. Decrease `batch_size`
2. Use fewer `n_batches` but larger batch size
3. Check that CIFAR100 download succeeded in `--data_dir`

## Advanced: Custom Layer Selection

To implement adaptive layer selection based on training progress:

```python
# In validate.py, inside training loop:
for i, (batch, targets) in enumerate(train_loader):
    def loss_fn():
        return criterion(model(batch), targets).item()
    
    # Progressive curriculum
    if i < n_batches // 3:
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    elif i < 2 * n_batches // 3:
        optimizer.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
    else:
        optimizer.layer_names = ["layer4.0.conv1.weight", "layer4.1.conv2.weight", "fc.weight", "fc.bias"]
    
    loss = optimizer.step(loss_fn)
```

## Expected Performance

With SPSA + good augmentation + fine head initialization:

- **Batch 32, Steps 32** (~64 fwd passes): ~20-30% accuracy
- **Batch 16, Steps 64** (~128 fwd passes): ~25-35% accuracy
- **Batch 8, Steps 128** (~256 fwd passes): ~30-40% accuracy

(These are rough estimates; actual results depend on hyperparameter choices)

## Files to Reference

- **Main optimizer**: `zo_optimizer.py` - SPSA implementation
- **Data transforms**: `augmentation.py` - Training augmentations
- **Head init**: `head_init.py` - Classification head initialization
- **Validator**: `validate.py` - Official evaluation (don't modify)
- **Model**: `model.py` - ResNet18 architecture (don't modify)

---

**Next step**: Run validate.py and experiment with different hyperparameters to find the best balance for your setup!
