# ================================================================
# Task 1: Supervised Baseline — CPU Friendly Version
# What we're doing: Training a ResNet-18 from scratch using
# only 10% of labeled CIFAR-10 data, to set a baseline.
# On CPU with 200 images and 5 epochs: ~3-5 minutes to run.
# ================================================================

# --- Import libraries ---
# Think of libraries as toolboxes. We import specific toolboxes
# for specific jobs.

import torch                         # PyTorch: the main deep learning engine
import torch.nn as nn                # nn = neural network building blocks
import torchvision                   # ready-made datasets and model architectures
import torchvision.transforms as T   # image transformation tools
from torch.utils.data import DataLoader, Subset
# DataLoader: feeds data in batches during training
# Subset: lets us use only specific indices from a dataset

import numpy as np                   # math/array operations
import matplotlib                    # plotting library
matplotlib.use('Agg')                # use non-interactive backend (works without display)
import matplotlib.pyplot as plt      # the actual plotting functions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# confusion_matrix: shows which classes the model confuses
import os                            # for creating folders

# ================================================================
# SECTION 1: Setup — Reproducibility and Device
# ================================================================

# Fix the random seed: this makes sure results are the same
# every time you run the code. The assignment requires seed=2026.
SEED = 2026
torch.manual_seed(SEED)          # controls PyTorch's random numbers
np.random.seed(SEED)             # controls NumPy's random numbers
torch.backends.cudnn.deterministic = True  # makes GPU deterministic (no effect on CPU)

# Automatically use GPU if available, otherwise use CPU.
# On your machine this will say "Using: cpu" — that is fine.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================================
# SECTION 2: Image Transforms (Preprocessing)
# ================================================================

# WHY normalize? Neural networks train better when input values
# are small and centered around 0. These specific mean/std values
# are pre-calculated from the entire CIFAR-10 dataset.

# Training transforms: add randomness to artificially increase variety
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    # Pad image by 4 pixels on each side then randomly crop back to 32x32
    # This shifts the image slightly, helping the model learn position invariance

    T.RandomHorizontalFlip(),
    # 50% chance to mirror the image horizontally
    # A car facing left is still a car

    T.ToTensor(),
    # Convert PIL image (H x W x C, values 0-255)
    # to PyTorch tensor (C x H x W, values 0.0-1.0)

    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 channel means (R, G, B)
        std=(0.2470, 0.2435, 0.2616)    # CIFAR-10 channel std deviations
    )
    # Normalize: output = (input - mean) / std
    # After this, most values are between -2 and +2
])

# Evaluation transforms: NO random changes — we want consistent results
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

# ================================================================
# SECTION 3: Load Dataset and Apply Fixed Splits
# ================================================================

# Load the full CIFAR-10 training and test datasets
# torchvision downloads them automatically (~170MB, once only)
print("\nLoading CIFAR-10 dataset (downloads once, then cached)...")

full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data',           # save dataset in a 'data' subfolder
    train=True,              # True = training portion (50,000 images)
    download=True,           # download if not already there
    transform=train_transform
)

full_test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,             # False = test portion (10,000 images)
    download=True,
    transform=eval_transform
)

# Helper function to read the .txt split files
def load_indices(filepath):
    """
    Opens a split file, reads each line as an integer index.
    Returns a list like [0, 5, 12, 33, ...] — the image indices to use.
    """
    with open(filepath, 'r') as f:
        indices = []
        for line in f:
            line = line.strip()   # remove spaces/newlines
            if line:              # skip empty lines
                indices.append(int(line))
    return indices

# Load our fixed split files
labeled_indices = load_indices('splits/train_labeled_10percent.txt')
val_indices     = load_indices('splits/val.txt')
test_indices    = load_indices('splits/test.txt')

print(f"Labeled training images: {len(labeled_indices)}")
print(f"Validation images: {len(val_indices)}")
print(f"Test images: {len(test_indices)}")

# Subset creates a "view" into the full dataset using only our indices
# It does NOT copy the data — it just remembers which indices to use
train_dataset = Subset(full_train_dataset, labeled_indices)
val_dataset   = Subset(full_test_dataset,  val_indices)
test_dataset  = Subset(full_test_dataset,  test_indices)

# ================================================================
# SECTION 4: DataLoaders
# ================================================================

# DataLoader handles:
# - Batching: grouping images into batches of N
# - Shuffling: random order each epoch (only for training)
# - Loading: can load data in parallel (num_workers)

# CPU-friendly settings: batch_size=32, num_workers=0
# (num_workers=0 means load data in main process — safer on Windows)
BATCH_SIZE = 32   # process 32 images at a time
                  # (use 64 if you have GPU; 32 is safer for CPU)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      # shuffle training data each epoch
    num_workers=0      # 0 = safest for Windows CPU users
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # no need to shuffle evaluation data
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ================================================================
# SECTION 5: Build the Model
# ================================================================

# ResNet-18 is a well-known neural network with 18 layers.
# It was designed for ImageNet (224x224 images).
# We modify it for CIFAR-10 (32x32 images).

def build_resnet18_cifar():
    """
    Build a ResNet-18 modified for CIFAR-10:
    - Replace first 7x7 conv with 3x3 (32x32 images are too small for 7x7)
    - Remove maxpool (would shrink 32x32 too aggressively)
    - Replace final layer: 1000 ImageNet classes → 10 CIFAR-10 classes
    """
    # Load ResNet-18 architecture with RANDOM weights
    # weights=None means NO pretrained ImageNet weights (required by assignment)
    model = torchvision.models.resnet18(weights=None)

    # MODIFICATION 1: Replace the first convolution
    # Original: kernel_size=7, stride=2 (designed for 224x224 images)
    # New: kernel_size=3, stride=1 (better for 32x32 images)
    model.conv1 = nn.Conv2d(
        in_channels=3,    # 3 color channels (RGB)
        out_channels=64,  # produces 64 feature maps (standard ResNet)
        kernel_size=3,    # 3x3 convolution window
        stride=1,         # slide 1 pixel at a time (no downsampling)
        padding=1,        # add 1 pixel of zeros around border to preserve size
        bias=False        # no bias (batch norm handles this)
    )

    # MODIFICATION 2: Remove max pooling
    # Original maxpool would turn 32x32 → 16x16 (too small)
    # nn.Identity() just passes the tensor through unchanged
    model.maxpool = nn.Identity()

    # MODIFICATION 3: Replace final classification layer
    # ResNet-18 originally outputs 1000 classes (ImageNet)
    # We need 10 classes (CIFAR-10)
    # in_features=512: the 512-dimensional feature vector before classification
    model.fc = nn.Linear(in_features=512, out_features=10)

    return model

# Create the model and move it to our device (CPU in your case)
model = build_resnet18_cifar().to(device)

# Count how many learnable parameters the model has
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel has {total_params:,} parameters")

# ================================================================
# SECTION 6: Loss Function and Optimizer
# ================================================================

# CrossEntropyLoss: standard loss for multi-class classification
# It measures "how wrong" our predictions are vs the true labels
# Lower = better predictions
criterion = nn.CrossEntropyLoss()

# Adam optimizer: updates the model's weights to reduce the loss
# lr = learning rate: how big each update step is
# 3e-4 = 0.0003 (small steps → stable training)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# ================================================================
# SECTION 7: Training Loop
# ================================================================

# CPU-friendly: use only 5 epochs
# (full assignment needs 30+ but that would take hours on CPU)
# For testing/development 5 epochs is enough to see it working
NUM_EPOCHS = 5

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
print(f"Each epoch processes {len(train_loader)} batches of {BATCH_SIZE} images")
print("(This will take a few minutes on CPU — please wait)\n")

train_losses = []   # store training loss for each epoch
val_losses   = []   # store validation loss for each epoch
val_accs     = []   # store validation accuracy for each epoch

for epoch in range(NUM_EPOCHS):

    # ── TRAINING PHASE ──────────────────────────────────────
    # model.train() enables training-specific behaviors:
    # - Batch normalization: updates running statistics
    # - Dropout: randomly zeros some neurons (regularization)
    model.train()

    running_loss = 0.0   # accumulate loss over all batches
    n_batches    = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: tensor of shape (batch_size, 3, 32, 32)
        # labels: tensor of shape (batch_size,) with values 0-9

        # Move data to same device as model
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Zero out gradients from previous batch
        # If we don't do this, gradients accumulate incorrectly
        optimizer.zero_grad()

        # Step 2: Forward pass — feed images through network
        # outputs: shape (batch_size, 10) — score for each class
        outputs = model(images)

        # Step 3: Compute loss — how wrong are our predictions?
        loss = criterion(outputs, labels)

        # Step 4: Backward pass — compute gradients
        # This figures out how to change each weight to reduce loss
        loss.backward()

        # Step 5: Update weights using the gradients
        optimizer.step()

        running_loss += loss.item()   # .item() extracts the Python float
        n_batches    += 1

        # Print progress every 5 batches
        if (batch_idx + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / n_batches
    train_losses.append(avg_train_loss)

    # ── VALIDATION PHASE ────────────────────────────────────
    # model.eval() disables training behaviors (batch norm, dropout)
    model.eval()

    val_loss   = 0.0
    val_correct = 0
    val_total   = 0

    # torch.no_grad(): don't compute gradients during evaluation
    # This saves memory and speeds up computation
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()

            # Get the predicted class for each image
            # torch.max returns (values, indices) — we want indices
            _, predicted = torch.max(outputs, dim=1)

            # Count correct predictions
            val_correct += (predicted == labels).sum().item()
            val_total   += labels.size(0)   # labels.size(0) = batch size

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = val_correct / val_total

    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} SUMMARY:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Val Acc:    {val_acc:.4f}  ({val_correct}/{val_total} correct)\n")

print("Training complete!")

# ================================================================
# SECTION 8: Plot and Save Loss Curves
# ================================================================

os.makedirs('graphs', exist_ok=True)   # create graphs/ folder if needed

epochs_list = list(range(1, NUM_EPOCHS + 1))

plt.figure(figsize=(8, 20))
plt.plot(epochs_list, train_losses, 'b-o', label='Training Loss',   linewidth=2, markersize=5)
plt.plot(epochs_list, val_losses,   'r-o', label='Validation Loss', linewidth=2, markersize=5)
plt.xlabel('Epoch',        fontsize=12)
plt.ylabel('Loss',         fontsize=12)
plt.title('Task 1 — Supervised ResNet-18 Training (10% Labels)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/supervised_loss.png', dpi=150)
plt.close()   # close the figure to free memory
print("Saved: graphs/supervised_loss.png")

# ================================================================
# SECTION 9: Test Set Evaluation
# ================================================================

print("\nEvaluating on test set...")
model.eval()
all_preds  = []   # collect all predictions
all_labels = []   # collect all true labels

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

# Convert to numpy arrays for sklearn
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute overall accuracy
test_accuracy = np.mean(all_preds == all_labels)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print("(Low accuracy is expected with only 5 epochs and small dataset)")

# ================================================================
# SECTION 10: Confusion Matrix
# ================================================================

# Class names for CIFAR-10 (in order, index 0-9)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Compute confusion matrix
# cm[i][j] = number of images of class i predicted as class j
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'Supervised Baseline — Confusion Matrix\nTest Accuracy: {test_accuracy*100:.1f}%')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/supervised_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/supervised_confusion_matrix.png")

# ================================================================
# SECTION 11: Save the model
# ================================================================

os.makedirs('models', exist_ok=True)
# state_dict() = the model's learned weights (not the architecture)
torch.save(model.state_dict(), 'models/supervised_baseline.pt')
print("Saved: models/supervised_baseline.pt")

# ================================================================
# SECTION 12: Print summary for your report
# ================================================================

print("\n" + "="*50)
print("COPY THESE NUMBERS INTO YOUR REPORT:")
print("="*50)
print(f"Model: Supervised ResNet-18, 10% labels")
print(f"Epochs trained: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training samples: {len(labeled_indices)}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print("="*50)