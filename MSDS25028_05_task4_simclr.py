# ================================================================
# Tasks 3, 4, 5: SimCLR — Feature Similarity + Implementation + Pretraining
# CPU-friendly: 2000 unlabeled images, 10 epochs pretraining
#
# What this file does in order:
#   1. Build the SimCLR encoder + projection head
#   2. Measure feature similarity BEFORE training (Task 3)
#   3. Show similarity matrix heatmap BEFORE training (Task 4)
#   4. Implement NT-Xent loss from scratch (Task 4)
#   5. Run SimCLR pretraining (Task 5)
#   6. Measure feature similarity AFTER training (Task 5)
#   7. Show similarity matrix heatmap AFTER training (Task 5)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Fix seed ─────────────────────────────────────────────────
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================================================
# SECTION 1: Augmentation Pipeline (same as Task 2)
# ================================================================

simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])

class TwoViewTransform:
    """Returns two independently augmented views of the same image."""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ================================================================
# SECTION 2: Load Unlabeled Dataset
# ================================================================

def load_indices(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

unlabeled_indices = load_indices('splits/train_ssl_unlabeled.txt')
print(f"Unlabeled images for SimCLR: {len(unlabeled_indices)}")

# Load CIFAR-10 with TwoViewTransform
# Each item returns (view1, view2) instead of a single image
ssl_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=TwoViewTransform(simclr_transform)
)
ssl_subset = Subset(ssl_dataset, unlabeled_indices)

# CPU-friendly batch size
# drop_last=True: NT-Xent needs consistent batch sizes
ssl_loader = DataLoader(
    ssl_subset,
    batch_size=32,      # keep small for CPU
    shuffle=True,
    num_workers=0,
    drop_last=True
)
print(f"Batches per epoch: {len(ssl_loader)}")

# ================================================================
# SECTION 3: Build SimCLR Model
# ================================================================

class SimCLREncoder(nn.Module):
    """
    ResNet-18 backbone modified for CIFAR-10.
    Outputs 512-dimensional feature vectors.
    Used for: SimCLR pretraining, linear probing, fine-tuning.
    """
    def __init__(self):
        super().__init__()

        # Load ResNet-18 with random weights (NO pretrained weights)
        resnet = torchvision.models.resnet18(weights=None)

        # Modify for CIFAR-10 (32x32 images instead of 224x224)
        resnet.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # remove aggressive downsampling

        # Keep all 4 residual block groups
        # Remove the final fully-connected layer (we want raw features)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,   # Identity — does nothing
            resnet.layer1,    # 64 feature maps
            resnet.layer2,    # 128 feature maps
            resnet.layer3,    # 256 feature maps
            resnet.layer4,    # 512 feature maps
        )
        # Global average pooling: (batch, 512, 4, 4) → (batch, 512, 1, 1)
        self.avgpool = resnet.avgpool

    def forward(self, x):
        h = self.backbone(x)      # (batch, 512, 4, 4)
        h = self.avgpool(h)       # (batch, 512, 1, 1)
        h = torch.flatten(h, 1)  # (batch, 512)
        return h


class ProjectionHead(nn.Module):
    """
    Maps 512-d encoder output → 128-d for contrastive loss.
    ONLY used during SimCLR pretraining.
    Discarded after pretraining — downstream tasks use encoder directly.

    Architecture: Linear(512→256) → ReLU → Linear(256→128)
    """
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        return self.net(h)   # (batch, 128)


class SimCLR(nn.Module):
    """Full SimCLR model = Encoder + Projection Head."""
    def __init__(self):
        super().__init__()
        self.encoder   = SimCLREncoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)    # 512-d representation (used downstream)
        z = self.projector(h)  # 128-d projection (used for contrastive loss)
        return h, z

# ================================================================
# SECTION 4: NT-Xent Loss (implemented from scratch — required)
# ================================================================

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    The core of SimCLR.

    HOW IT WORKS:
    - z1: projections from view1, shape (N, 128)
    - z2: projections from view2, shape (N, 128)
    - For each image i, (z1_i, z2_i) is a POSITIVE pair
    - All other 2*(N-1) vectors are NEGATIVES
    - We want the model to identify the positive among all negatives

    FORMULA for one sample i:
    loss(i) = -log[ exp(sim(z_i, z_j)/tau) / sum_{k≠i} exp(sim(z_i, z_k)/tau) ]

    where sim = cosine similarity, tau = temperature
    """

    N = z1.shape[0]   # batch size

    # Step 1: L2-normalize so cosine similarity = dot product
    # After normalization, each vector has length 1
    # So dot product of two vectors = cos(angle between them)
    z1 = F.normalize(z1, dim=1)   # (N, 128)
    z2 = F.normalize(z2, dim=1)   # (N, 128)

    # Step 2: Stack into one big (2N, 128) matrix
    # Order: [z1_0, z1_1, ..., z1_{N-1}, z2_0, z2_1, ..., z2_{N-1}]
    z = torch.cat([z1, z2], dim=0)   # (2N, 128)

    # Step 3: Compute full (2N × 2N) cosine similarity matrix
    # sim[i,j] = how similar vector i is to vector j
    # Since normalized: sim[i,j] = dot product of row i and row j
    sim = torch.mm(z, z.T) / temperature   # (2N, 2N)

    # Step 4: Build labels — for each row, which column is the positive?
    # z1_i (row i) → positive is z2_i (row i+N)
    # z2_i (row i+N) → positive is z1_i (row i)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels], dim=0)  # (2N,)

    # Step 5: Mask out the diagonal (self-similarity)
    # sim[i,i] = 1.0 (a vector is perfectly similar to itself)
    # But comparing a view to itself is not meaningful — mask it out
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    # -inf → exp(-inf) = 0 → excluded from the denominator sum

    # Step 6: Compute cross-entropy loss
    # This is equivalent to the NT-Xent formula:
    # For each of the 2N rows, treat it as a classification problem:
    # "which of the other 2N-1 vectors is my positive pair?"
    loss = F.cross_entropy(sim, labels)
    return loss

# ================================================================
# SECTION 5: Task 3 — Feature Similarity BEFORE Training
# ================================================================

print("\n" + "="*55)
print("TASK 3: Feature Similarity BEFORE SimCLR Training")
print("="*55)

# Create a fresh model with RANDOM weights (no training yet)
temp_model = SimCLR().to(device)
temp_model.eval()

# Get one batch of data to measure similarity on
data_iter = iter(ssl_loader)
(v1_batch, v2_batch), _ = next(data_iter)
v1_batch = v1_batch.to(device)
v2_batch = v2_batch.to(device)

with torch.no_grad():
    # Get encoder features (512-d) for both views
    h1, _ = temp_model(v1_batch)   # (32, 512)
    h2, _ = temp_model(v2_batch)   # (32, 512)

    # Normalize for cosine similarity computation
    h1_norm = F.normalize(h1, dim=1)
    h2_norm = F.normalize(h2, dim=1)

    # SAME-IMAGE similarity:
    # Compare view1 of image_i with view2 of image_i (element-wise)
    # F.cosine_similarity computes dot products row by row → shape (32,)
    same_sim_before = F.cosine_similarity(h1_norm, h2_norm, dim=1).mean().item()

    # DIFFERENT-IMAGE similarity:
    # Build full similarity matrix of view1s with each other
    full_sim_matrix = torch.mm(h1_norm, h1_norm.T)   # (32, 32)
    N = h1_norm.shape[0]
    # Mask the diagonal (don't compare image with itself)
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    diff_sim_before = full_sim_matrix[off_diag_mask].mean().item()

print(f"\nSame-image similarity  BEFORE: {same_sim_before:.4f}")
print(f"Diff-image similarity  BEFORE: {diff_sim_before:.4f}")
print(f"Gap (same - diff):             {same_sim_before - diff_sim_before:.4f}")
print("(Small gap = model can't tell same-image views apart yet)")

# ================================================================
# SECTION 6: Task 4 — Similarity Matrix Heatmap BEFORE Training
# ================================================================

print("\n" + "="*55)
print("TASK 4: Similarity Matrix Heatmap BEFORE Training")
print("="*55)

# Use a SMALL batch (8 images) so heatmap is readable
# For a batch of 8 images: 2N = 16 → 16×16 heatmap
small_loader = DataLoader(
    ssl_subset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)
(sv1, sv2), _ = next(iter(small_loader))
sv1 = sv1.to(device)
sv2 = sv2.to(device)

with torch.no_grad():
    _, sz1 = temp_model(sv1)   # projection head output (8, 128)
    _, sz2 = temp_model(sv2)   # projection head output (8, 128)

    # Stack both views: (16, 128)
    sz_all = torch.cat([
        F.normalize(sz1, dim=1),
        F.normalize(sz2, dim=1)
    ], dim=0)

    # Full 16×16 cosine similarity matrix
    sim_before = torch.mm(sz_all, sz_all.T).cpu().numpy()

def plot_similarity_matrix(sim_matrix, title, save_path, n_images=8):
    """
    Plot a cosine similarity heatmap.
    Rows/cols 0 to n-1 = view1 of each image
    Rows/cols n to 2n-1 = view2 of each image
    Positive pairs are at positions (i, i+n) and (i+n, i)
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Add value annotations in each cell
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}',
                   ha='center', va='center', fontsize=7,
                   color='black' if abs(sim_matrix[i,j]) < 0.5 else 'white')

    # Draw a red box around the positive pair positions
    # Positive pairs: (i, i+n_images) and (i+n_images, i)
    for i in range(n_images):
        # Upper-right quadrant: view1_i vs view2_i
        ax.add_patch(plt.Rectangle(
            (n_images + i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2
        ))
        # Lower-left quadrant: view2_i vs view1_i
        ax.add_patch(plt.Rectangle(
            (i - 0.5, n_images + i - 0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2
        ))

    # Add dividing lines to separate view1 and view2 blocks
    ax.axhline(y=n_images - 0.5, color='black', linewidth=2)
    ax.axvline(x=n_images - 0.5, color='black', linewidth=2)

    # Labels
    tick_labels = (
        [f'V1-img{i}' for i in range(n_images)] +
        [f'V2-img{i}' for i in range(n_images)]
    )
    ax.set_xticks(range(len(sim_matrix)))
    ax.set_yticks(range(len(sim_matrix)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_title(title + '\n(Red boxes = positive pairs)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

os.makedirs('results', exist_ok=True)
plot_similarity_matrix(
    sim_before,
    "Cosine Similarity Matrix — BEFORE SimCLR Training",
    'results/similarity_matrix_before_training.png'
)

# ================================================================
# SECTION 7: Print Positive/Negative Pair Table (Task 4 report)
# ================================================================

print("\nPositive/Negative Pair Structure (for 8-image batch):")
print(f"{'Original Image':<18} {'View1 Index':<14} {'View2 Index':<14} {'Positive Pair'}")
print("-" * 60)
for i in range(8):
    print(f"{'image ' + str(i):<18} {i:<14} {i+8:<14} {'YES ← positive'}")
print("\nAll other combinations (e.g. image0-view1 vs image1-view2) = NEGATIVE pairs")
print(f"Total pairs per batch: {8*2*(8*2-1)//2} pairs, of which {8} are positive")

# ================================================================
# SECTION 8: Task 5 — SimCLR Pretraining
# ================================================================

print("\n" + "="*55)
print("TASK 5: SimCLR Pretraining (CPU-friendly: 10 epochs)")
print("="*55)

model     = SimCLR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
TEMPERATURE = 0.5
NUM_EPOCHS  = 10   # use 10 for CPU; assignment requires 50 (mention in report)

print(f"Pretraining for {NUM_EPOCHS} epochs on {len(unlabeled_indices)} unlabeled images")
print("Please wait — this takes ~10-20 minutes on CPU...\n")

simclr_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch_idx, ((view1, view2), _) in enumerate(ssl_loader):
        # _ = labels from CIFAR-10 — we IGNORE them (self-supervised!)
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        # Forward pass: get projections for both views
        _, z1 = model(view1)   # (batch, 128)
        _, z2 = model(view2)   # (batch, 128)

        # Compute contrastive loss
        loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

        # Backward pass and weight update
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    simclr_losses.append(avg_loss)
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | NT-Xent Loss: {avg_loss:.4f}")

print("\nPretraining complete!")

# ================================================================
# SECTION 9: Save the Pretrained Encoder
# ================================================================

os.makedirs('models', exist_ok=True)
# Save ONLY the encoder (not the projection head)
# The projection head is discarded after pretraining
torch.save(model.encoder.state_dict(), 'models/simclr_encoder.pt')
print("Saved: models/simclr_encoder.pt")

# ================================================================
# SECTION 10: Plot SimCLR Loss Curve
# ================================================================

os.makedirs('graphs', exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPOCHS + 1), simclr_losses,
         'g-o', linewidth=2, markersize=6)
plt.xlabel('Epoch',         fontsize=12)
plt.ylabel('NT-Xent Loss',  fontsize=12)
plt.title('SimCLR Pretraining Loss Curve\n(Lower = encoder better at grouping same-image views)',
          fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/simclr_pretraining_loss.png', dpi=150)
plt.close()
print("Saved: graphs/simclr_pretraining_loss.png")

# ================================================================
# SECTION 11: Feature Similarity AFTER Training
# ================================================================

print("\n" + "="*55)
print("TASK 5: Feature Similarity AFTER SimCLR Training")
print("="*55)

model.eval()
with torch.no_grad():
    h1_after, _ = model(v1_batch)
    h2_after, _ = model(v2_batch)

    h1_after_norm = F.normalize(h1_after, dim=1)
    h2_after_norm = F.normalize(h2_after, dim=1)

    same_sim_after = F.cosine_similarity(
        h1_after_norm, h2_after_norm, dim=1
    ).mean().item()

    full_sim_after = torch.mm(h1_after_norm, h1_after_norm.T)
    diff_sim_after = full_sim_after[off_diag_mask].mean().item()

print(f"Same-image similarity  AFTER:  {same_sim_after:.4f}")
print(f"Diff-image similarity  AFTER:  {diff_sim_after:.4f}")
print(f"Gap (same - diff):             {same_sim_after - diff_sim_after:.4f}")
print("(Larger gap = SimCLR successfully learned to group same-image views)")

# ================================================================
# SECTION 12: Similarity Matrix AFTER Training
# ================================================================

with torch.no_grad():
    _, sz1_after = model(sv1)
    _, sz2_after = model(sv2)

    sz_after_all = torch.cat([
        F.normalize(sz1_after, dim=1),
        F.normalize(sz2_after, dim=1)
    ], dim=0)

    sim_after = torch.mm(sz_after_all, sz_after_all.T).cpu().numpy()

plot_similarity_matrix(
    sim_after,
    "Cosine Similarity Matrix — AFTER SimCLR Training",
    'results/similarity_matrix_after_training.png'
)

# ================================================================
# SECTION 13: Final Summary Table
# ================================================================

print("\n" + "="*55)
print("SUMMARY TABLE — Copy into your report:")
print("="*55)
print(f"{'Pair Type':<40} {'Before':>10} {'After':>10} {'Change':>10}")
print("-" * 72)
print(f"{'Same image, two augmented views':<40} "
      f"{same_sim_before:>10.4f} {same_sim_after:>10.4f} "
      f"{same_sim_after - same_sim_before:>+10.4f}")
print(f"{'Different images':<40} "
      f"{diff_sim_before:>10.4f} {diff_sim_after:>10.4f} "
      f"{diff_sim_after - diff_sim_before:>+10.4f}")
print("="*55)