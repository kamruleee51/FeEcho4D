# ======================
# Training / DataLoader options
# ======================
shuffle_     = False   # Shuffle training data each epoch? (handled by custom samplers/loaders elsewhere)
shuffle_val  = False   # Shuffle validation data? Typically False for reproducibility

trainBatch = 4         # Training batch size (number of volumes per batch)
testBatch  = 1         # Test batch size (keep small to avoid OOM and preserve order)
valBatch   = 4         # Validation batch size

# Temporal windowing for NeighborTimeBatchSampler
window_size = 4        # Number of consecutive time frames per batch window
stride      = 4        # Step between windows (== window_size → non-overlapping)

# ======================
# Optimization schedule
# ======================
num_epochs = 100       # Total training epochs
LR         = 1e-4      # Initial learning rate (Adam)

# ======================
# Data / I/O
# ======================
img_size   = 256       # Input image size (H = W = 256 after transforms)
num_workers = 0        # DataLoader workers (0 for full determinism / Windows compatibility)

# ======================
# Data augmentation bounds (used elsewhere)
# ======================
lower_bound = 0.999    # Lower bound for certain augment/probability thresholds
upper_bound = 1.99     # Upper bound for certain augment/probability thresholds

# ======================
# Model / task config
# ======================
num_classes  = 3       # Number of segmentation classes (e.g., background, MYO, LV)
val_interval = 1       # Validate every N epochs (1 = validate each epoch)
