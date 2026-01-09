from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("models/fire_model.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    
    return result, round(conf_percent, 2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="No file uploaded")
    
    file = request.files["file"]
    
    if file.filename == "":
        return render_template("index.html", result="No file selected")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    result, confidence = predict_fire(filepath)
    
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    print("=" * 50)
    print("Fire Detection System")
    print("=" * 50)
    print("Open browser: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)