# ================================================================
# MSDS25028_05_allCode.py
# Deep Learning Spring 2026 — Assignment 5: SimCLR
# Student: Farwa
# Roll Number: MSDS25028
# Seed: 2026
#
# This file contains ALL tasks in one place:
# Task 0  — Conceptual answers (as comments)
# Task 1  — Supervised baseline
# Task 2  — Augmentation pipeline + TwoViewTransform
# Task 3  — Feature similarity before training
# Task 4  — SimCLR implementation (encoder + projection + NT-Xent)
# Task 5  — SimCLR pretraining
# Task 6  — Linear probe evaluation
# Task 7  — Fine-tuning
# Task 8  — t-SNE visualization
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
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import csv
import os

# ================================================================
# GLOBAL SETUP — Seed, Device, Folders
# ================================================================

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs('graphs',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ================================================================
# TASK 0 — Conceptual Warm-up (answers written as comments)
# ================================================================

# Q1. In supervised classification, what does the model learn from?
#     The model learns from labeled images. Each image is paired
#     with a class label and the model adjusts weights to predict
#     that label correctly using cross-entropy loss.

# Q2. Do unlabeled images still contain useful structure?
#     Yes. Even without labels, images contain edges, textures,
#     shapes and color patterns. A cat image still looks like a cat.
#     Labels are just human-assigned names — visual content remains.

# Q3. Should two augmented views of the same image be similar?
#     Yes. Both views come from the same original image and depict
#     the same object. A good encoder should map them to nearby
#     points in feature space.

# Q4. Should features from different images be similar?
#     No. Images from different originals have different semantic
#     content. Their feature vectors should be far apart in
#     representation space.

# Q5. Why is a model trained without labels still useful?
#     Because it learns general visual features — shapes, textures,
#     object structure — that transfer well to downstream tasks.
#     SimCLR learns what things look like without learning names.

# Q6. What is the difference between pretraining and fine-tuning?
#     Pretraining = training on a large general task first (SimCLR
#     on unlabeled data) to build a feature extractor.
#     Fine-tuning = continuing training on the specific downstream
#     task (classification with labels) using pretrained weights.

print("\n" + "="*60)
print("TASK 0: Conceptual warm-up — see comments in code")
print("="*60)

# ================================================================
# SHARED UTILITIES
# ================================================================

def load_indices(filepath):
    """Read split file and return list of integer indices."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Standard evaluation transform (no random augmentation)
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])

# Load split indices
labeled_indices   = load_indices('splits/train_labeled_10percent.txt')
unlabeled_indices = load_indices('splits/train_ssl_unlabeled.txt')
val_indices       = load_indices('splits/val.txt')
test_indices      = load_indices('splits/test.txt')

print(f"\nDataset splits loaded:")
print(f"  Unlabeled (SimCLR) : {len(unlabeled_indices)}")
print(f"  Labeled (10%)      : {len(labeled_indices)}")
print(f"  Validation         : {len(val_indices)}")
print(f"  Test               : {len(test_indices)}")

# Load full CIFAR-10 datasets
full_train = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=True, transform=eval_transform)
full_test  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=eval_transform)

# Create labeled dataloaders
train_loader = DataLoader(Subset(full_train, labeled_indices),
                          batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(Subset(full_test,  val_indices),
                          batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(Subset(full_test,  test_indices),
                          batch_size=32, shuffle=False, num_workers=0)

# ================================================================
# SHARED MODEL ARCHITECTURE
# ================================================================

class SimCLREncoder(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 (32x32 images).
    Changes from standard ResNet-18:
    - conv1: 7x7 stride2 → 3x3 stride1 (preserves small image size)
    - maxpool: removed (replaced with Identity)
    - fc: removed (we want raw 512-d features)
    Output: 512-dimensional feature vector
    """
    def __init__(self):
        super().__init__()
        resnet         = torchvision.models.resnet18(weights=None)
        resnet.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone  = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.avgpool = resnet.avgpool

    def forward(self, x):
        h = self.backbone(x)     # (batch, 512, 1, 1)
        h = self.avgpool(h)      # (batch, 512, 1, 1)
        return torch.flatten(h, 1)  # (batch, 512)


class ProjectionHead(nn.Module):
    """
    Maps 512-d encoder output → 128-d for contrastive loss.
    Architecture: Linear(512→256) → ReLU → Linear(256→128)
    Used ONLY during SimCLR pretraining. Discarded afterward.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, h):
        return self.net(h)


class SimCLR(nn.Module):
    """Full SimCLR model = Encoder + Projection Head."""
    def __init__(self):
        super().__init__()
        self.encoder   = SimCLREncoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)    # 512-d (used downstream)
        z = self.projector(h)  # 128-d (used for contrastive loss)
        return h, z

# ================================================================
# TASK 1 — Supervised Baseline
# ================================================================

print("\n" + "="*60)
print("TASK 1: Supervised Baseline (ResNet-18, 10% labels)")
print("="*60)

# Build supervised model
def build_supervised_model():
    model    = torchvision.models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

sup_model  = build_supervised_model().to(device)
sup_optim  = torch.optim.Adam(sup_model.parameters(), lr=3e-4)
criterion  = nn.CrossEntropyLoss()
SUP_EPOCHS = 15

sup_train_losses = []
sup_val_losses   = []

for epoch in range(SUP_EPOCHS):
    # Training
    sup_model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        sup_optim.zero_grad()
        loss = criterion(sup_model(images), labels)
        loss.backward()
        sup_optim.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    sup_train_losses.append(avg_train)

    # Validation
    sup_model.eval()
    val_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out  = sup_model(images)
            val_loss += criterion(out, labels).item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

    avg_val = val_loss / len(val_loader)
    sup_val_losses.append(avg_val)
    print(f"  Epoch {epoch+1:02d}/{SUP_EPOCHS} | "
          f"Train Loss: {avg_train:.4f} | "
          f"Val Loss: {avg_val:.4f} | "
          f"Val Acc: {correct/total:.4f}")

# Test accuracy
sup_model.eval()
sup_preds = []
sup_labels_all = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, pred = torch.max(sup_model(images), 1)
        sup_preds.extend(pred.cpu().numpy().tolist())
        sup_labels_all.extend(labels.numpy().tolist())

supervised_acc = np.mean(np.array(sup_preds) == np.array(sup_labels_all))
print(f"\nTask 1 — Test Accuracy: {supervised_acc:.4f} ({supervised_acc*100:.1f}%)")

# Save loss plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, SUP_EPOCHS+1), sup_train_losses,
         'b-o', linewidth=2, markersize=5, label='Train Loss')
plt.plot(range(1, SUP_EPOCHS+1), sup_val_losses,
         'r-o', linewidth=2, markersize=5, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Task 1 — Supervised ResNet-18 Loss')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('graphs/supervised_loss.png', dpi=150)
plt.close()
print("Saved: graphs/supervised_loss.png")

# Save confusion matrix
cm   = confusion_matrix(sup_labels_all, sup_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'Supervised Baseline Confusion Matrix '
             f'(Acc: {supervised_acc*100:.1f}%)')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('results/supervised_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/supervised_confusion_matrix.png")

torch.save(sup_model.state_dict(), 'models/supervised_baseline.pt')

# ================================================================
# TASK 2 — Augmentation Pipeline + TwoViewTransform
# ================================================================

print("\n" + "="*60)
print("TASK 2: Augmentation Pipeline")
print("="*60)

simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])

class TwoViewTransform:
    """
    Applies the same augmentation pipeline twice independently.
    Returns two different augmented views of the same image.
    This is the core of SimCLR's self-supervised signal.
    """
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

def denormalize(tensor):
    """Reverse normalization for display."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)

# Load raw CIFAR-10 for visualization
raw_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=None)

n_examples = 10
fig, axes  = plt.subplots(nrows=n_examples, ncols=3,
                           figsize=(9, 2.5*n_examples))

for row in range(n_examples):
    pil_img, label = raw_dataset[row]
    v1 = simclr_transform(pil_img)
    v2 = simclr_transform(pil_img)

    orig_np = np.array(pil_img) / 255.0
    v1_np   = denormalize(v1).permute(1,2,0).numpy()
    v2_np   = denormalize(v2).permute(1,2,0).numpy()

    axes[row,0].imshow(orig_np)
    axes[row,0].set_title(f"Original ({CLASS_NAMES[label]})",
                          fontsize=9)
    axes[row,0].axis('off')

    axes[row,1].imshow(v1_np)
    axes[row,1].set_title("Augmented View 1", fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].imshow(v2_np)
    axes[row,2].set_title("Augmented View 2", fontsize=9)
    axes[row,2].axis('off')

plt.suptitle("SimCLR: Two Augmented Views from One Image",
             fontsize=12, y=1.002)
plt.tight_layout()
plt.savefig('results/augmentation_examples.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/augmentation_examples.png")

# ================================================================
# TASK 3 — Feature Similarity BEFORE Training
# ================================================================

print("\n" + "="*60)
print("TASK 3: Feature Similarity BEFORE SimCLR Training")
print("="*60)

# Load unlabeled data with TwoViewTransform
ssl_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=TwoViewTransform(simclr_transform)
)
ssl_subset = Subset(ssl_dataset, unlabeled_indices)
ssl_loader = DataLoader(ssl_subset, batch_size=32,
                        shuffle=True, num_workers=0, drop_last=True)

# Random encoder — no training yet
temp_model = SimCLR().to(device)
temp_model.eval()

(v1_batch, v2_batch), _ = next(iter(ssl_loader))
v1_batch = v1_batch.to(device)
v2_batch = v2_batch.to(device)

with torch.no_grad():
    h1, _ = temp_model(v1_batch)
    h2, _ = temp_model(v2_batch)
    h1_n  = F.normalize(h1, dim=1)
    h2_n  = F.normalize(h2, dim=1)

    same_sim_before = F.cosine_similarity(h1_n, h2_n, dim=1).mean().item()
    full_sim        = torch.mm(h1_n, h1_n.T)
    N               = h1_n.shape[0]
    off_diag        = ~torch.eye(N, dtype=torch.bool, device=device)
    diff_sim_before = full_sim[off_diag].mean().item()

print(f"Same-image similarity  BEFORE: {same_sim_before:.4f}")
print(f"Diff-image similarity  BEFORE: {diff_sim_before:.4f}")
print(f"Gap (same - diff):             {same_sim_before-diff_sim_before:.4f}")

# ================================================================
# TASK 4 — SimCLR Implementation
# ================================================================

print("\n" + "="*60)
print("TASK 4: SimCLR Implementation")
print("="*60)

# ── NT-Xent Loss (implemented from scratch) ──────────────────
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent Contrastive Loss — implemented from scratch.

    For each image i, (z1_i, z2_i) is the positive pair.
    All other 2*(N-1) vectors are negatives.

    Formula:
    loss(i,j) = -log[ exp(sim(zi,zj)/tau) /
                      sum_{k!=i} exp(sim(zi,zk)/tau) ]

    Steps:
    1. L2-normalize projections
    2. Stack into 2N matrix
    3. Compute 2N x 2N similarity matrix
    4. Mask diagonal (self-similarity)
    5. Build labels pointing each row to its positive
    6. Apply cross-entropy over all 2N rows
    """
    N  = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)            # (2N, 128)

    sim = torch.mm(z, z.T) / temperature        # (2N, 2N)

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])    # (2N,)

    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    return F.cross_entropy(sim, labels)

# ── Similarity Matrix Heatmap BEFORE Training ────────────────
small_loader = DataLoader(ssl_subset, batch_size=8,
                          shuffle=False, num_workers=0)
(sv1, sv2), _ = next(iter(small_loader))
sv1, sv2 = sv1.to(device), sv2.to(device)

with torch.no_grad():
    _, sz1 = temp_model(sv1)
    _, sz2 = temp_model(sv2)
    sz_all = torch.cat([F.normalize(sz1, dim=1),
                        F.normalize(sz2, dim=1)], dim=0)
    sim_before = torch.mm(sz_all, sz_all.T).cpu().numpy()

def plot_sim_matrix(sim_matrix, title, save_path, n=8):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}',
                   ha='center', va='center', fontsize=7,
                   color='black' if abs(sim_matrix[i,j])<0.5
                   else 'white')
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (n+i-0.5, i-0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2))
        ax.add_patch(plt.Rectangle(
            (i-0.5, n+i-0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2))
    ax.axhline(y=n-0.5, color='black', linewidth=2)
    ax.axvline(x=n-0.5, color='black', linewidth=2)
    ticks = ([f'V1-{i}' for i in range(n)] +
             [f'V2-{i}' for i in range(n)])
    ax.set_xticks(range(len(sim_matrix)))
    ax.set_yticks(range(len(sim_matrix)))
    ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_title(title + '\n(Red = positive pairs)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

plot_sim_matrix(sim_before,
                "Similarity Matrix — BEFORE SimCLR Training",
                'results/similarity_matrix_before_training.png')

# Print positive/negative pair table
print("\nPositive/Negative Pair Table (8-image batch):")
print(f"{'Original Image':<18} {'View1 Index':<14} "
      f"{'View2 Index':<14} {'Positive Pair'}")
print("-"*60)
for i in range(8):
    print(f"{'image '+str(i):<18} {i:<14} {i+8:<14} YES")
print(f"\nTotal pairs: 120 | Positive pairs: 8 | "
      f"Negative pairs: 112")

# ================================================================
# TASK 5 — SimCLR Pretraining
# ================================================================

print("\n" + "="*60)
print("TASK 5: SimCLR Pretraining")
print("="*60)

model       = SimCLR().to(device)
optimizer   = torch.optim.Adam(model.parameters(), lr=3e-4)
TEMPERATURE = 0.5
SSL_EPOCHS  = 15

print(f"Training for {SSL_EPOCHS} epochs on "
      f"{len(unlabeled_indices)} unlabeled images...")

simclr_losses = []

for epoch in range(SSL_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for (view1, view2), _ in ssl_loader:
        # Labels ignored — self-supervised!
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss  = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    avg = epoch_loss / n_batches
    simclr_losses.append(avg)
    print(f"  Epoch {epoch+1:02d}/{SSL_EPOCHS} | "
          f"NT-Xent Loss: {avg:.4f}")

# Save encoder
torch.save(model.encoder.state_dict(), 'models/simclr_encoder.pt')
print("Saved: models/simclr_encoder.pt")

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, SSL_EPOCHS+1), simclr_losses,
         'g-o', linewidth=2, markersize=6)
plt.xlabel('Epoch'); plt.ylabel('NT-Xent Loss')
plt.title('Task 5 — SimCLR Pretraining Loss')
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('graphs/simclr_pretraining_loss.png', dpi=150)
plt.close()
print("Saved: graphs/simclr_pretraining_loss.png")

# Feature similarity AFTER training
model.eval()
with torch.no_grad():
    h1a, _ = model(v1_batch)
    h2a, _ = model(v2_batch)
    h1a_n  = F.normalize(h1a, dim=1)
    h2a_n  = F.normalize(h2a, dim=1)

    same_sim_after = F.cosine_similarity(
        h1a_n, h2a_n, dim=1).mean().item()
    full_sim_after = torch.mm(h1a_n, h1a_n.T)
    diff_sim_after = full_sim_after[off_diag].mean().item()

print(f"\nSame-image similarity  AFTER:  {same_sim_after:.4f}")
print(f"Diff-image similarity  AFTER:  {diff_sim_after:.4f}")
print(f"Gap (same - diff):             "
      f"{same_sim_after-diff_sim_after:.4f}")

# Similarity matrix AFTER training
with torch.no_grad():
    _, sz1a = model(sv1)
    _, sz2a = model(sv2)
    sz_all_a = torch.cat([F.normalize(sz1a, dim=1),
                           F.normalize(sz2a, dim=1)], dim=0)
    sim_after = torch.mm(sz_all_a, sz_all_a.T).cpu().numpy()

plot_sim_matrix(sim_after,
                "Similarity Matrix — AFTER SimCLR Training",
                'results/similarity_matrix_after_training.png')

print("\nSummary Table:")
print(f"{'Pair Type':<40} {'Before':>10} {'After':>10}")
print("-"*62)
print(f"{'Same image, two augmented views':<40} "
      f"{same_sim_before:>10.4f} {same_sim_after:>10.4f}")
print(f"{'Different images':<40} "
      f"{diff_sim_before:>10.4f} {diff_sim_after:>10.4f}")

# ================================================================
# TASK 6 — Linear Probe Evaluation
# ================================================================

print("\n" + "="*60)
print("TASK 6: Linear Probe Evaluation")
print("="*60)

def run_linear_probe(encoder, name, epochs=15):
    """Freeze encoder, train only Linear(512→10)."""
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    clf       = nn.Linear(512, 10).to(device)
    opt       = torch.optim.Adam(clf.parameters(), lr=3e-4)
    crit      = nn.CrossEntropyLoss()
    val_accs  = []

    print(f"\n── {name} ──")
    for epoch in range(epochs):
        clf.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            with torch.no_grad():
                feats = encoder(images)
            loss = crit(clf(feats), labels)
            loss.backward(); opt.step()

        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, pred = torch.max(clf(encoder(images)), 1)
                correct += (pred==labels).sum().item()
                total   += labels.size(0)
        val_accs.append(correct/total)
        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Val Acc: {correct/total:.4f}")

    # Test
    clf.eval()
    correct = total = 0
    preds_all = []; labels_all = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = clf(encoder(images))
            _, pred = torch.max(out, 1)
            correct    += (pred==labels).sum().item()
            total      += labels.size(0)
            preds_all.extend(pred.cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

    test_acc = correct/total
    print(f"  [{name}] Test Accuracy: {test_acc:.4f}")
    return test_acc, val_accs, clf, preds_all, labels_all

# Experiment A — Random encoder
random_enc = SimCLREncoder().to(device)
(acc_random, val_accs_random,
 _, _, _) = run_linear_probe(random_enc, "Random Encoder")

# Experiment B — SimCLR encoder
simclr_enc = SimCLREncoder().to(device)
simclr_enc.load_state_dict(
    torch.load('models/simclr_encoder.pt', map_location=device))
(acc_simclr, val_accs_simclr,
 linear_head, _, _) = run_linear_probe(
    simclr_enc, "SimCLR Encoder")

# Save linear probe model
torch.save({'encoder': simclr_enc.state_dict(),
            'classifier': linear_head.state_dict()},
           'models/linear_probe.pt')
print("Saved: models/linear_probe.pt")

# Plot
epochs_lp = list(range(1, len(val_accs_random)+1))
plt.figure(figsize=(9, 5))
plt.plot(epochs_lp, val_accs_random, '--', color='gray',
         linewidth=2, marker='o', markersize=5,
         label='Random Encoder (val)')
plt.plot(epochs_lp, val_accs_simclr, '-', color='green',
         linewidth=2, marker='o', markersize=5,
         label='SimCLR Encoder (val)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Task 6 — Linear Probe Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3)
plt.ylim(0, 1); plt.tight_layout()
plt.savefig('graphs/linear_probe_accuracy.png', dpi=150)
plt.close()
print("Saved: graphs/linear_probe_accuracy.png")

print(f"\nRandom encoder test accuracy : {acc_random:.4f}")
print(f"SimCLR encoder test accuracy : {acc_simclr:.4f}")
print(f"Improvement from SimCLR      : "
      f"+{(acc_simclr-acc_random)*100:.1f}%")

# ================================================================
# TASK 7 — Fine-tuning
# ================================================================

print("\n" + "="*60)
print("TASK 7: Fine-tuning SimCLR Encoder")
print("="*60)

ft_encoder    = SimCLREncoder().to(device)
ft_encoder.load_state_dict(
    torch.load('models/simclr_encoder.pt', map_location=device))
ft_classifier = nn.Linear(512, 10).to(device)

# Unfreeze encoder — full end-to-end training
for p in ft_encoder.parameters():
    p.requires_grad = True

ft_optim   = torch.optim.Adam(
    list(ft_encoder.parameters()) +
    list(ft_classifier.parameters()), lr=3e-4)
FT_EPOCHS  = 15
ft_val_accs = []

print(f"Fine-tuning for {FT_EPOCHS} epochs (encoder NOT frozen)...")

for epoch in range(FT_EPOCHS):
    ft_encoder.train(); ft_classifier.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        ft_optim.zero_grad()
        loss = criterion(ft_classifier(ft_encoder(images)), labels)
        loss.backward(); ft_optim.step()

    ft_encoder.eval(); ft_classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = torch.max(
                ft_classifier(ft_encoder(images)), 1)
            correct += (pred==labels).sum().item()
            total   += labels.size(0)
    ft_val_accs.append(correct/total)
    print(f"  Epoch {epoch+1:02d}/{FT_EPOCHS} | "
          f"Val Acc: {correct/total:.4f}")

# Test accuracy
ft_encoder.eval(); ft_classifier.eval()
correct = total = 0
ft_preds = []; ft_labels = []; ft_probs = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out   = ft_classifier(ft_encoder(images))
        probs = torch.softmax(out, dim=1)
        _, pred = torch.max(out, 1)
        correct    += (pred==labels).sum().item()
        total      += labels.size(0)
        ft_preds.extend(pred.cpu().numpy().tolist())
        ft_labels.extend(labels.cpu().numpy().tolist())
        ft_probs.extend(probs.cpu().numpy().tolist())

acc_finetune = correct/total
print(f"\nFine-tuning Test Accuracy: {acc_finetune:.4f} "
      f"({acc_finetune*100:.1f}%)")

# Save model
torch.save({'encoder': ft_encoder.state_dict(),
            'classifier': ft_classifier.state_dict()},
           'models/finetuned_model.pt')
print("Saved: models/finetuned_model.pt")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ft_val_accs)+1), ft_val_accs,
         'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Task 7 — Fine-tuning Validation Accuracy')
plt.grid(True, alpha=0.3); plt.ylim(0, 1); plt.tight_layout()
plt.savefig('graphs/finetuning_accuracy.png', dpi=150)
plt.close()
print("Saved: graphs/finetuning_accuracy.png")

# Save test_predictions.csv
with open('results/test_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['image_index','true_label','predicted_label']
    header += [f'prob_class_{i}' for i in range(10)]
    writer.writerow(header)
    for idx, (true, pred, probs) in enumerate(
            zip(ft_labels, ft_preds, ft_probs)):
        writer.writerow(
            [test_indices[idx], true, pred] + probs)
print("Saved: results/test_predictions.csv")

# Save metrics.json
metrics = {
    "student_name"                     : "Farwa",
    "roll_number"                      : "MSDS25028",
    "seed"                             : 2026,
    "batch_size"                       : 32,
    "simclr_epochs"                    : SSL_EPOCHS,
    "linear_probe_epochs"              : 15,
    "finetuning_epochs"                : FT_EPOCHS,
    "learning_rate"                    : 0.0003,
    "temperature"                      : 0.5,
    "supervised_10percent_test_acc"    : round(supervised_acc, 4),
    "random_linear_probe_test_acc"     : round(acc_random, 4),
    "simclr_linear_probe_test_acc"     : round(acc_simclr, 4),
    "simclr_finetune_test_acc"         : round(acc_finetune, 4),
    "same_view_similarity_before"      : round(same_sim_before, 4),
    "different_image_similarity_before": round(diff_sim_before, 4),
    "same_view_similarity_after"       : round(same_sim_after, 4),
    "different_image_similarity_after" : round(diff_sim_after, 4),
    "note": "CPU only. Reduced dataset: 3000 unlabeled, "
            "800 labeled, 500 val, 500 test. 15 epochs each."
}
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Saved: results/metrics.json")

# ================================================================
# TASK 8 — t-SNE Visualization
# ================================================================

print("\n" + "="*60)
print("TASK 8: t-SNE Feature Visualization")
print("="*60)

vis_indices = val_indices[:200]
vis_loader  = DataLoader(
    Subset(full_test, vis_indices),
    batch_size=32, shuffle=False, num_workers=0)

COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#bfef45','#fabed4','#469990']

def extract_features(encoder):
    encoder.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for images, labels in vis_loader:
            feats.append(encoder(images.to(device)).cpu().numpy())
            lbls.append(labels.numpy())
    return np.vstack(feats), np.concatenate(lbls)

def plot_tsne(features, labels, title, save_path):
    print(f"  Running t-SNE: {title}")
    tsne   = TSNE(n_components=2,
                  perplexity=min(50, len(features)//4),
                  random_state=SEED)
    coords = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in range(10):
        mask = labels == cls
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=COLORS[cls], label=CLASS_NAMES[cls],
                   alpha=0.7, s=20, edgecolors='none')
    ax.legend(markerscale=2, fontsize=9, ncol=2, framealpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# 1. Random encoder
rand_enc_vis = SimCLREncoder().to(device)
feats, lbls  = extract_features(rand_enc_vis)
plot_tsne(feats, lbls,
          "t-SNE: Random Encoder Features",
          'results/random_encoder_pca_or_tsne.png')

# 2. SimCLR encoder
simclr_enc_vis = SimCLREncoder().to(device)
simclr_enc_vis.load_state_dict(
    torch.load('models/simclr_encoder.pt', map_location=device))
feats, lbls = extract_features(simclr_enc_vis)
plot_tsne(feats, lbls,
          "t-SNE: SimCLR Pretrained Encoder Features",
          'results/simclr_encoder_pca_or_tsne.png')

# 3. Fine-tuned encoder
ft_enc_vis = SimCLREncoder().to(device)
ckpt = torch.load('models/finetuned_model.pt', map_location=device)
ft_enc_vis.load_state_dict(ckpt['encoder'])
feats, lbls = extract_features(ft_enc_vis)
plot_tsne(feats, lbls,
          "t-SNE: Fine-tuned Encoder Features",
          'results/finetuned_encoder_pca_or_tsne.png')

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<45} {'Test Acc':>10}")
print("-"*57)
print(f"{'Supervised ResNet-18, 10% labels':<45} "
      f"{supervised_acc:>10.4f}")
print(f"{'Random encoder + linear probe':<45} "
      f"{acc_random:>10.4f}")
print(f"{'SimCLR encoder + linear probe':<45} "
      f"{acc_simclr:>10.4f}")
print(f"{'SimCLR encoder + fine-tuning':<45} "
      f"{acc_finetune:>10.4f}")
print("="*60)
print("\nAll files saved successfully!")
print("metrics.json and test_predictions.csv are in results/")
print("All plots are in graphs/ and results/")