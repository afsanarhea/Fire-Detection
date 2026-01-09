import os
from PIL import Image

INPUT_SIZE = (224, 224)

fire_input = "data/raw/fire_images"
non_fire_input = "data/raw/non_fire_images"

fire_output = "data/processed/fire"
non_fire_output = "data/processed/no_fire"


os.makedirs(fire_output, exist_ok=True)
os.makedirs(non_fire_output, exist_ok=True)

def resize_images(input_path, output_path, label):
    images = os.listdir(input_path)
    count = 0
    
    for img_name in images:
        try:
            img_path = os.path.join(input_path, img_name)
            img = Image.open(img_path)
            img = img.convert("RGB")
            img = img.resize(INPUT_SIZE)
            
            
            save_path = os.path.join(output_path, img_name)
            img.save(save_path)
            count += 1
        except Exception as e:
            print(f"Error: {img_name} - {e}")
    
    print(f"{label}: {count} images processed")


print("Processing images...")
print("=" * 40)
resize_images(fire_input, fire_output, "Fire")
resize_images(non_fire_input, non_fire_output, "No Fire")
print("=" * 40)
print("All images resized to 224x224!")
