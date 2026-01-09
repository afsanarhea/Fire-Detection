import os
from PIL import Image
import matplotlib.pyplot as plt

fire_path = "data/raw/fire_images"
non_fire_path = "data/raw/non_fire_images"

fire_count = len(os.listdir(fire_path))
non_fire_count = len(os.listdir(non_fire_path))

print("=" * 40)
print("Fire Detection - Dataset Info")
print("=" * 40)
print(f"Fire images: {fire_count}")
print(f"Non-fire images: {non_fire_count}")
print(f"Total images: {fire_count + non_fire_count}")
print("=" * 40)

fire_images = os.listdir(fire_path)[:3]
non_fire_images = os.listdir(non_fire_path)[:3]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, img_name in enumerate(fire_images):
    img = Image.open(os.path.join(fire_path, img_name))
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"FIRE")
    axes[0, i].axis("off")

for i, img_name in enumerate(non_fire_images):
    img = Image.open(os.path.join(non_fire_path, img_name))
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"NO FIRE")
    axes[1, i].axis("off")

plt.suptitle("Fire Detection Dataset - Fire vs Non-Fire", fontsize=16)
plt.tight_layout()
plt.savefig("data/sample_images.png")
plt.show()

print("\nSample images saved to: data/sample_images.png")