# ================================================================
# Task 6: Linear Probe Evaluation
# 
# WHAT WE ARE DOING:
# We freeze the encoder (no weight updates) and train ONLY
# a single linear layer on top. If SimCLR learned good features,
# even this simple linear classifier should do well.
#
# TWO EXPERIMENTS:
# A) Random encoder (untrained) + linear head
# B) SimCLR pretrained encoder + linear head
#
# Comparing A vs B tells us exactly how much SimCLR helped.
# CPU friendly: 500 labeled images, 15 epochs
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================================================
# SECTION 1: Rebuild Encoder Architecture
# ================================================================

class SimCLREncoder(nn.Module):
    """
    ResNet-18 modified for CIFAR-10.
    Outputs 512-dimensional feature vectors.
    This is the SAME architecture used in task4_simclr.py
    We reload it here with the pretrained weights.
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone  = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.avgpool = resnet.avgpool

    def forward(self, x):
        h = self.backbone(x)
        h = self.avgpool(h)
        return torch.flatten(h, 1)   # (batch, 512)

# ================================================================
# SECTION 2: Load Datasets with Fixed Splits
# ================================================================

def load_indices(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

labeled_indices = load_indices('splits/train_labeled_10percent.txt')
val_indices     = load_indices('splits/val.txt')
test_indices    = load_indices('splits/test.txt')

print(f"Labeled training images : {len(labeled_indices)}")
print(f"Validation images       : {len(val_indices)}")
print(f"Test images             : {len(test_indices)}")

# Standard eval transform — no random augmentation for evaluation
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
])

full_train = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=True, transform=eval_transform)
full_test  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=eval_transform)

train_loader = DataLoader(Subset(full_train, labeled_indices),
                          batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(Subset(full_test,  val_indices),
                          batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(Subset(full_test,  test_indices),
                          batch_size=32, shuffle=False, num_workers=0)

# ================================================================
# SECTION 3: Linear Probe Training Function
# ================================================================

def run_linear_probe(encoder, encoder_name, num_epochs=15):
    """
    Freeze the encoder and train only a Linear(512 -> 10) head.

    freeze = True means:
    - Encoder weights NEVER change during this training
    - Only the linear head weights are updated
    - This is called 'linear probing'

    If the encoder has learned good features, even this simple
    linear classifier should achieve reasonable accuracy.
    """

    # FREEZE the encoder — no gradients will flow through it
    for param in encoder.parameters():
        param.requires_grad = False
    # requires_grad=False means PyTorch will NOT compute or update
    # gradients for these parameters during backward()

    encoder.eval()   # set to eval mode (freezes batchnorm too)
    encoder = encoder.to(device)

    # Simple linear classification head
    # Input: 512-d feature vector from encoder
    # Output: 10 class scores
    classifier = nn.Linear(512, 10).to(device)

    # Only optimize classifier parameters (encoder is frozen)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs   = []

    print(f"\n── {encoder_name} Linear Probe ──")

    for epoch in range(num_epochs):

        # ── Training ──
        classifier.train()
        correct = total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Extract features from frozen encoder
            # torch.no_grad() here because encoder is frozen
            # (saves memory and time — no need to store gradients)
            with torch.no_grad():
                features = encoder(images)   # (batch, 512)

            # Only the classifier does a forward+backward pass
            outputs = classifier(features)   # (batch, 10)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total
        train_accs.append(train_acc)

        # ── Validation ──
        classifier.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)
                features = encoder(images)
                outputs  = classifier(features)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        print(f"  Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ── Test Accuracy ──
    classifier.eval()
    test_correct = test_total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images   = images.to(device)
            labels   = labels.to(device)
            features = encoder(images)
            outputs  = classifier(features)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_acc = test_correct / test_total
    print(f"\n  [{encoder_name}] Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    return test_acc, train_accs, val_accs, classifier, all_preds, all_labels

# ================================================================
# SECTION 4: Experiment A — Random Encoder Linear Probe
# ================================================================

print("\n" + "="*55)
print("EXPERIMENT A: Random Encoder (untrained) Linear Probe")
print("="*55)

random_encoder = SimCLREncoder().to(device)
# No weights loaded — completely random initialization
# This tells us: how much does the encoder architecture
# itself help, even without any training?

(test_acc_random,
 train_accs_random,
 val_accs_random,
 _, _, _) = run_linear_probe(random_encoder, "Random Encoder")

# ================================================================
# SECTION 5: Experiment B — SimCLR Pretrained Encoder Linear Probe
# ================================================================

print("\n" + "="*55)
print("EXPERIMENT B: SimCLR Pretrained Encoder Linear Probe")
print("="*55)

simclr_encoder = SimCLREncoder().to(device)

# Load the weights saved during SimCLR pretraining
simclr_encoder.load_state_dict(
    torch.load('models/simclr_encoder.pt', map_location=device)
)
print("Loaded SimCLR pretrained weights from models/simclr_encoder.pt")

(test_acc_simclr,
 train_accs_simclr,
 val_accs_simclr,
 linear_head,
 test_preds_simclr,
 test_labels_simclr) = run_linear_probe(simclr_encoder, "SimCLR Encoder")

# ================================================================
# SECTION 6: Save Linear Probe Model
# ================================================================

os.makedirs('models', exist_ok=True)
torch.save({
    'encoder_state_dict'   : simclr_encoder.state_dict(),
    'classifier_state_dict': linear_head.state_dict()
}, 'models/linear_probe.pt')
print("\nSaved: models/linear_probe.pt")

# ================================================================
# SECTION 7: Plot Accuracy Comparison
# ================================================================

os.makedirs('graphs', exist_ok=True)
epochs_list = list(range(1, len(val_accs_random) + 1))

plt.figure(figsize=(9, 5))
plt.plot(epochs_list, val_accs_random,
         '--', color='gray',  linewidth=2,
         label='Random Encoder (val acc)',  marker='o', markersize=5)
plt.plot(epochs_list, val_accs_simclr,
         '-',  color='green', linewidth=2,
         label='SimCLR Encoder (val acc)', marker='o', markersize=5)

plt.xlabel('Epoch',    fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Linear Probe — Validation Accuracy\nRandom Encoder vs SimCLR Encoder',
          fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('graphs/linear_probe_accuracy.png', dpi=150)
plt.close()
print("Saved: graphs/linear_probe_accuracy.png")

# ================================================================
# SECTION 8: Print Summary
# ================================================================

print("\n" + "="*55)
print("LINEAR PROBE RESULTS :")
print("="*55)
print(f"{'Model':<35} {'Encoder':<25} {'Test Accuracy':>15}")
print("-"*75)
print(f"{'Random Linear Probe':<35} {'Random frozen':<25} {test_acc_random:>14.4f}")
print(f"{'SimCLR Linear Probe':<35} {'SimCLR frozen':<25} {test_acc_simclr:>14.4f}")
print(f"\nImprovement from SimCLR: +{(test_acc_simclr - test_acc_random)*100:.1f}%")
print("="*55)