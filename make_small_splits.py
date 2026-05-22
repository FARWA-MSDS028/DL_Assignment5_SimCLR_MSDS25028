# This script creates SMALL split files 
# Instead of 45,000 images, we use 500. Instead of 5,000 val, we use 200.
import os
os.makedirs('splits', exist_ok=True)

# Write 500 indices for unlabeled SSL training (images 0 to 499)
with open('splits/train_ssl_unlabeled.txt', 'w') as f:
    for i in range(500):
        f.write(f"{i}\n")

# Write 200 indices for labeled 10% training (images 500 to 699)
with open('splits/train_labeled_10percent.txt', 'w') as f:
    for i in range(500, 700):
        f.write(f"{i}\n")

# Write 100 indices for validation (images 0 to 99 from test set)
with open('splits/val.txt', 'w') as f:
    for i in range(100):
        f.write(f"{i}\n")

# Write 100 indices for test (images 100 to 199 from test set)
with open('splits/test.txt', 'w') as f:
    for i in range(100, 200):
        f.write(f"{i}\n")

print("Small split files created in splits/ folder!")
print("  train_ssl_unlabeled.txt  → 500 images")
print("  train_labeled_10percent.txt → 200 images")
print("  val.txt  → 100 images")
print("  test.txt → 100 images")