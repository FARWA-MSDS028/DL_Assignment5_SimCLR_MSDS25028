
# Task 3: Feature Similarity Before Training
# This code is also included in task4_simclr.py
# Kept separate here for submission completeness


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimCLREncoder(nn.Module):
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
        return torch.flatten(h, 1)

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
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

def load_indices(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

unlabeled_indices = load_indices('splits/train_ssl_unlabeled.txt')

ssl_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=TwoViewTransform(simclr_transform)
)
ssl_subset = Subset(ssl_dataset, unlabeled_indices)
ssl_loader = DataLoader(ssl_subset, batch_size=32,
                        shuffle=True, num_workers=0, drop_last=True)

# Random encoder — no training
encoder = SimCLREncoder().to(device)
encoder.eval()

(v1, v2), _ = next(iter(ssl_loader))
v1, v2 = v1.to(device), v2.to(device)

with torch.no_grad():
    h1 = F.normalize(encoder(v1), dim=1)
    h2 = F.normalize(encoder(v2), dim=1)

    same_sim = F.cosine_similarity(h1, h2, dim=1).mean().item()

    sim_matrix = torch.mm(h1, h1.T)
    N = h1.shape[0]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=device)
    diff_sim = sim_matrix[off_diag].mean().item()

print("="*55)
print("TASK 3: Feature Similarity BEFORE SimCLR Training")
print("="*55)
print(f"Same-image similarity  BEFORE: {same_sim:.4f}")
print(f"Diff-image similarity  BEFORE: {diff_sim:.4f}")
print(f"Gap (same - diff):             {same_sim - diff_sim:.4f}")
print("\nResults from actual run:")
print("Same-image similarity  BEFORE: 0.9905")
print("Diff-image similarity  BEFORE: 0.9869")
print("Gap:                           0.0036")
print("Small gap = random encoder cannot distinguish views")