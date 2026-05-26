# This script creates SMALL split files 
# Instead of 45,000 images, we use 500. Instead of 5,000 val, we use 200.
import os
os.makedirs('splits', exist_ok=True)

# Unlabeled data for SimCLR pretraining (increased: 500 → 2000)
with open('splits/train_ssl_unlabeled.txt', 'w') as f:
    for i in range(2000):
        f.write(f"{i}\n")

# Labeled 10% for supervised/finetuning (increased: 200 → 500)
with open('splits/train_labeled_10percent.txt', 'w') as f:
    for i in range(2000, 2500):   # starts after unlabeled so no overlap
        f.write(f"{i}\n")

# Validation (increased: 100 → 300)
with open('splits/val.txt', 'w') as f:
    for i in range(300):
        f.write(f"{i}\n")

# Test (increased: 100 → 300, different indices from val)
with open('splits/test.txt', 'w') as f:
    for i in range(300, 600):
        f.write(f"{i}\n")

print("Split files created!")
print(f"  train_ssl_unlabeled.txt     → 2000 images")
print(f"  train_labeled_10percent.txt → 500 images")
print(f"  val.txt                     → 300 images")
print(f"  test.txt                    → 300 images")