import os
import shutil
import random

TRAIN_RATIO = 0.8 

fire_input = "data/processed/fire"
no_fire_input = "data/processed/no_fire"

train_fire = "data/processed/train/fire"
train_no_fire = "data/processed/train/no_fire"
test_fire = "data/processed/test/fire"
test_no_fire = "data/processed/test/no_fire"

for folder in [train_fire, train_no_fire, test_fire, test_no_fire]:
    os.makedirs(folder, exist_ok=True)

def split_data(input_path, train_path, test_path, label):
    images = os.listdir(input_path)
    random.shuffle(images)
    
    split_point = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_point]
    test_images = images[split_point:]
    
    for img in train_images:
        src = os.path.join(input_path, img)
        dst = os.path.join(train_path, img)
        shutil.copy(src, dst)
    
    for img in test_images:
        src = os.path.join(input_path, img)
        dst = os.path.join(test_path, img)
        shutil.copy(src, dst)
    
    print(f"{label}: Train={len(train_images)}, Test={len(test_images)}")

print("Splitting data...")
print("=" * 40)
split_data(fire_input, train_fire, test_fire, "Fire")
split_data(no_fire_input, train_no_fire, test_no_fire, "No Fire")
print("=" * 40)
print("Data split complete!")