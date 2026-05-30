# ================================================================
# Task 7: Fine-tuning the SimCLR Encoder
#
# WHAT WE ARE DOING:
# Unlike linear probing (encoder frozen), here we train the
# FULL model end-to-end — both encoder AND classifier update.
# We initialize the encoder with SimCLR pretrained weights
# and then fine-tune everything on the labeled data.
#
# This should give the best accuracy of all our experiments
# because the encoder can adapt its features to the task.
# CPU friendly: 500 labeled images, 15 epochs
# ================================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import csv
import os

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================================================
# SECTION 1: Encoder Architecture (same as before)
# ================================================================

class SimCLREncoder(nn.Module):
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
        h = self.backbone(x)
        h = self.avgpool(h)
        return torch.flatten(h, 1)

# ================================================================
# SECTION 2: Load Data
# ================================================================

def load_indices(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

labeled_indices = load_indices('splits/train_labeled_10percent.txt')
val_indices     = load_indices('splits/val.txt')
test_indices    = load_indices('splits/test.txt')

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
# SECTION 3: Fine-tuning Function
# ================================================================

def run_finetuning(encoder, classifier, experiment_name,
                   freeze_encoder=False, num_epochs=15):
    """
    Train encoder + classifier together (freeze_encoder=False)
    or just the classifier (freeze_encoder=True for linear probe).

    Fine-tuning = freeze_encoder=False
    Both encoder AND classifier weights update each step.
    The encoder adapts its features specifically for CIFAR-10.
    """

    for param in encoder.parameters():
        param.requires_grad = not freeze_encoder

    encoder    = encoder.to(device)
    classifier = classifier.to(device)

    # Optimize ALL parameters if fine-tuning
    if freeze_encoder:
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=3e-4)
    else:
        # Fine-tuning: optimize both encoder AND classifier
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) +
            list(classifier.parameters()), lr=3e-4)

    criterion = nn.CrossEntropyLoss()
    val_accs  = []

    print(f"\n── {experiment_name} ──")
    print(f"   Encoder frozen: {freeze_encoder}")

    for epoch in range(num_epochs):

        # Training
        if freeze_encoder:
            encoder.eval()
        else:
            encoder.train()   # encoder updates too
        classifier.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # If fine-tuning, gradients flow through encoder too
            features = encoder(images)
            outputs  = classifier(features)
            loss     = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        encoder.eval()
        classifier.eval()
        correct = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)
                outputs = classifier(encoder(images))
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        val_accs.append(val_acc)
        print(f"  Epoch {epoch+1:02d}/{num_epochs} | Val Acc: {val_acc:.4f}")

    # Test accuracy
    encoder.eval()
    classifier.eval()
    correct    = total = 0
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = classifier(encoder(images))
            probs   = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    test_acc = correct / total
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    return test_acc, val_accs, all_preds, all_labels, all_probs

# ================================================================
# SECTION 4: Run Fine-tuning Experiment
# ================================================================

print("\n" + "="*55)
print("TASK 7: Fine-tuning SimCLR Encoder")
print("="*55)

# Load SimCLR pretrained encoder
ft_encoder = SimCLREncoder().to(device)
ft_encoder.load_state_dict(
    torch.load('models/simclr_encoder.pt', map_location=device)
)
print("Loaded SimCLR pretrained weights")

# Fresh classification head
ft_classifier = nn.Linear(512, 10).to(device)

# Run fine-tuning (encoder NOT frozen — both update)
(test_acc_finetune,
 val_accs_finetune,
 ft_preds,
 ft_labels,
 ft_probs) = run_finetuning(
    ft_encoder, ft_classifier,
    "SimCLR Encoder + Full Fine-tuning",
    freeze_encoder=False,
    num_epochs=15
)

# ================================================================
# SECTION 5: Save Fine-tuned Model
# ================================================================

os.makedirs('models', exist_ok=True)
torch.save({
    'encoder_state_dict'   : ft_encoder.state_dict(),
    'classifier_state_dict': ft_classifier.state_dict()
}, 'models/finetuned_model.pt')
print("\nSaved: models/finetuned_model.pt")

# ================================================================
# SECTION 6: Plot Fine-tuning Accuracy
# ================================================================

os.makedirs('graphs', exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_accs_finetune) + 1), val_accs_finetune,
         'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch',    fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Fine-tuning — Validation Accuracy\n(SimCLR pretrained encoder, full end-to-end training)',
          fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('graphs/finetuning_accuracy.png', dpi=150)
plt.close()
print("Saved: graphs/finetuning_accuracy.png")

# ================================================================
# SECTION 7: Save test_predictions.csv
# ================================================================

os.makedirs('results', exist_ok=True)

with open('results/test_predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Header row
    header = ['image_index', 'true_label', 'predicted_label']
    header += [f'prob_class_{i}' for i in range(10)]
    writer.writerow(header)

    # One row per test image
    for idx, (true, pred, probs) in enumerate(
            zip(ft_labels, ft_preds, ft_probs)):
        row = [test_indices[idx], true, pred] + probs
        writer.writerow(row)

print("Saved: results/test_predictions.csv")

# ================================================================
# SECTION 8: Load previous results and save metrics.json
# ================================================================

# Read supervised baseline accuracy from earlier run
supervised_acc = 0.4560   # from Task 1 output

# Read linear probe results
# We load the numbers we printed from task5_linear_probe.py

random_probe_acc = 0.1820   
simclr_probe_acc = 0.4320   

# Similarity values from task4_simclr.py output
same_sim_before  = 0.9902
diff_sim_before  = 0.9858
same_sim_after   = 0.8820
diff_sim_after   = 0.5904

metrics = {
    "student_name"                   : "Farwa",
    "roll_number"                    : "MSDS25028",
    "seed"                           : 2026,
    "batch_size"                     : 32,
    "simclr_epochs"                  : 15,
    "linear_probe_epochs"            : 15,
    "finetuning_epochs"              : 15,
    "learning_rate"                  : 0.0003,
    "temperature"                    : 0.5,
    "supervised_10percent_test_acc"  : supervised_acc,
    "random_linear_probe_test_acc"   : random_probe_acc,
    "simclr_linear_probe_test_acc"   : simclr_probe_acc,
    "simclr_finetune_test_acc"       : test_acc_finetune,
    "same_view_similarity_before"    : same_sim_before,
    "different_image_similarity_before": diff_sim_before,
    "same_view_similarity_after"     : same_sim_after,
    "different_image_similarity_after" : diff_sim_after,
    "note": "Reduced dataset used due to CPU constraints"
}

with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Saved: results/metrics.json")

# ================================================================
# SECTION 9: Final Comparison Table
# ================================================================

print("\n" + "="*60)
print("FINAL COMPARISON TABLE :")
print("="*60)
print(f"{'Model':<40} {'Test Acc':>10}")
print("-"*52)
print(f"{'Supervised ResNet-18, 10% labels':<40} {supervised_acc:>10.4f}")
print(f"{'Random encoder + linear probe':<40} {random_probe_acc:>10.4f}")
print(f"{'SimCLR encoder + linear probe':<40} {simclr_probe_acc:>10.4f}")
print(f"{'SimCLR encoder + fine-tuning':<40} {test_acc_finetune:>10.4f}")
print("="*60)
print("\nAll files saved.")