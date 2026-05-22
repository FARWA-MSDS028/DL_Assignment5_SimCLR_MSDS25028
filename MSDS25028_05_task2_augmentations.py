# ================================================================
# Task 2: Augmentation Pipeline and Two-View Transform
# What we're doing: Showing how SimCLR creates two different
# "views" of each image using random transformations.
# ================================================================

import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Fix seed for reproducibility
torch.manual_seed(2026)
np.random.seed(2026)

# ================================================================
# SECTION 1: The SimCLR Augmentation Pipeline
# ================================================================

# This EXACT pipeline is specified in the assignment.
# Do NOT change any values. Each transform randomly distorts
# the image in a different way to create two distinct views.

simclr_transform = T.Compose([

    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    # WHAT: Randomly zoom in on a portion (20%-100%) of the image
    # and resize that portion back to 32x32.
    # WHY: Simulates seeing the same object from different distances.
    # Example: one view might see the full car, another just the hood.

    T.RandomHorizontalFlip(p=0.5),
    # WHAT: 50% chance to mirror the image left-right.
    # WHY: Whether a dog faces left or right doesn't change that it's a dog.

    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # WHAT: Randomly change brightness (how light/dark), contrast (light vs dark
    # difference), saturation (color intensity), hue (color shift).
    # Values: 0.4 means vary by up to ±40%
    # WHY: Same object looks different under different lighting conditions.

    T.RandomGrayscale(p=0.2),
    # WHAT: 20% chance to convert RGB to grayscale (but keep 3 channels).
    # WHY: Forces the model to rely on shapes/textures, not just color.

    T.ToTensor(),
    # WHAT: Convert PIL image to PyTorch tensor.
    # Specifically: (H, W, C) with values 0-255 → (C, H, W) with values 0.0-1.0

    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    # WHAT: Standardize pixel values using CIFAR-10's statistics.
    # After this: output = (pixel - mean) / std
    # WHY: Neural networks train more stably with normalized inputs.
])

# ================================================================
# SECTION 2: TwoViewTransform Class
# ================================================================

# The assignment requires us to implement this ourselves.
# It wraps any transform and applies it TWICE to the same image.
# Each application is RANDOM and INDEPENDENT → two different views.

class TwoViewTransform:
    """
    Takes a single image and returns two independently augmented versions.
    Used during SimCLR training so each image produces a positive pair.
    """
    def __init__(self, transform):
        # Store the transform pipeline
        self.transform = transform

    def __call__(self, x):
        # x = original PIL image (not yet transformed)
        view1 = self.transform(x)   # apply pipeline once → first augmented view
        view2 = self.transform(x)   # apply pipeline again → second augmented view
        # Because transforms are RANDOM, view1 ≠ view2
        # But both came from the same original x
        return view1, view2

# ================================================================
# SECTION 3: Load CIFAR-10 Without Transforms (Raw Images)
# ================================================================

# We load with transform=None to get raw PIL images.
# We'll apply transforms manually so we can show before/after.
print("Loading CIFAR-10...")
raw_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=None   # No transform! We want the raw PIL images.
)

# ================================================================
# SECTION 4: Helper Function — Denormalize for Display
# ================================================================

def denormalize(tensor):
    """
    Reverse the normalization so we can display the image.
    Normalization did: x_norm = (x - mean) / std
    Reverse is:        x = x_norm * std + mean
    Then clamp to [0,1] because pixel values must be in that range.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    x = tensor * std + mean        # reverse normalization
    x = torch.clamp(x, 0.0, 1.0)  # clip to valid range
    return x

# ================================================================
# SECTION 5: Visualize 10 Examples
# ================================================================

print("Generating augmentation visualization...")

n_examples = 10   # show 10 images

# Choose specific images (not random) for consistent results
# These indices are chosen to get a variety of classes
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create a grid: 10 rows, 3 columns (Original | View1 | View2)
fig, axes = plt.subplots(
    nrows=n_examples,
    ncols=3,
    figsize=(9, 2.5 * n_examples)   # width=9 inches, height scales with rows
)

for row_idx, img_idx in enumerate(sample_indices):

    # Get the original image (as PIL) and its label (integer 0-9)
    original_pil, label = raw_dataset[img_idx]

    # Apply augmentation pipeline ONCE to get View 1
    view1_tensor = simclr_transform(original_pil)

    # Apply augmentation pipeline AGAIN to get View 2 (different!)
    view2_tensor = simclr_transform(original_pil)

    # Convert original PIL image to numpy array for matplotlib
    # PIL image has shape (32, 32, 3) and values 0-255
    original_np = np.array(original_pil) / 255.0
    # Divide by 255 to get values 0.0-1.0 for matplotlib

    # Denormalize the view tensors and convert to numpy
    # tensor has shape (3, 32, 32) → we need (32, 32, 3) for matplotlib
    view1_np = denormalize(view1_tensor).permute(1, 2, 0).numpy()
    # .permute(1, 2, 0) rearranges dimensions: (C, H, W) → (H, W, C)
    view2_np = denormalize(view2_tensor).permute(1, 2, 0).numpy()

    # ── Column 0: Original Image ──
    axes[row_idx, 0].imshow(original_np)
    axes[row_idx, 0].set_title(
        f"Original\n({class_names[label]})",
        fontsize=9, pad=3
    )
    axes[row_idx, 0].axis('off')   # hide axis ticks and labels

    # ── Column 1: Augmented View 1 ──
    axes[row_idx, 1].imshow(view1_np)
    axes[row_idx, 1].set_title("Augmented View 1", fontsize=9, pad=3)
    axes[row_idx, 1].axis('off')

    # ── Column 2: Augmented View 2 ──
    axes[row_idx, 2].imshow(view2_np)
    axes[row_idx, 2].set_title("Augmented View 2", fontsize=9, pad=3)
    axes[row_idx, 2].axis('off')

# Add column headers
fig.text(0.18, 1.001, "Original Image", ha='center', fontsize=11, fontweight='bold')
fig.text(0.50, 1.001, "Augmented View 1", ha='center', fontsize=11, fontweight='bold')
fig.text(0.82, 1.001, "Augmented View 2", ha='center', fontsize=11, fontweight='bold')

plt.suptitle(
    "SimCLR Augmentation: Two Views from One Image\n"
    "Both views came from the same original — they form a positive pair",
    fontsize=12, y=1.005
)
plt.tight_layout()

# Save the figure
os.makedirs('results', exist_ok=True)
plt.savefig('results/augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/augmentation_examples.png")

# ================================================================
# SECTION 6: Verify TwoViewTransform works correctly
# ================================================================

print("\nVerifying TwoViewTransform...")

two_view = TwoViewTransform(simclr_transform)

# Test it on the first image
test_img, _ = raw_dataset[0]
v1, v2 = two_view(test_img)

print(f"  Input image type:  {type(test_img)}")
print(f"  View 1 shape: {v1.shape}  (C x H x W)")
print(f"  View 2 shape: {v2.shape}  (C x H x W)")
print(f"  Are views identical? {torch.allclose(v1, v2)}")
# Should print False — they're randomly different

# Check pixel difference between the two views
diff = (v1 - v2).abs().mean().item()
print(f"  Mean pixel difference between views: {diff:.4f}")
print("  (non-zero difference confirms views are different)")

print("\nTask 2 complete!")
print("\nReport answers:")
print("Q1. Are the two views identical? NO — each is randomly transformed")
print("Q2. Do they still show the same object? YES — same original image")
print("Q3. Why are they a positive pair? Because both came from same image")
print("Q4. Too weak augmentations? Model learns trivially, no useful features")
print("Q5. Too strong augmentations? Views look completely unrelated, model fails")