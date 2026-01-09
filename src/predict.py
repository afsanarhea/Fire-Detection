import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

model = load_model("models/fire_model.h5")

def predict_fire(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]
    
    if confidence < 0.5:
        result = "FIRE DETECTED"
        conf_percent = (1 - confidence) * 100
    else:
        result = "NO FIRE"
        conf_percent = confidence * 100
    
    return result, conf_percent

if __name__ == "__main__":
    print("=" * 50)
    print("Fire Detection - Prediction")
    print("=" * 50)
    
    test_fire = "data/processed/test/fire"
    test_no_fire = "data/processed/test/no_fire"
    
    import os
    
    fire_images = os.listdir(test_fire)
    if fire_images:
        img_path = os.path.join(test_fire, fire_images[0])
        result, conf = predict_fire(img_path)
        print(f"\nTest Image: {fire_images[0]}")
        print(f"Result: {result}")
        print(f"Confidence: {conf:.2f}%")
    
    no_fire_images = os.listdir(test_no_fire)
    if no_fire_images:
        img_path = os.path.join(test_no_fire, no_fire_images[0])
        result, conf = predict_fire(img_path)
        print(f"\nTest Image: {no_fire_images[0]}")
        print(f"Result: {result}")
        print(f"Confidence: {conf:.2f}%")
    
    print("=" * 50)