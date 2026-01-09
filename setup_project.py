import os

folders = [
    "data/raw/fire_images",
    "data/raw/non_fire_images",
    "data/processed/train/fire",
    "data/processed/train/no_fire",
    "data/processed/test/fire",
    "data/processed/test/no_fire",
    "notebooks",
    "models",
    "src"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\n Project structure ready!")