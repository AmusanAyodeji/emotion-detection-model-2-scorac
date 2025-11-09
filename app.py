import os
import sqlite3
from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np

MODEL_FILE = "model.h5"
DB_FILE = "emotion_users.db"

# 1. Train model if not already present
if not os.path.exists(MODEL_FILE):
    print("model.h5 not found, running model_training.py.")
    # Run your training script
    exit_code = os.system("python model_training.py")
    if exit_code != 0 or not os.path.exists(MODEL_FILE):
        raise RuntimeError("Error: Could not generate model.h5. Please check model_training.py and dataset.")

from tensorflow.keras.models import load_model
model = load_model(MODEL_FILE)

# Database setup
def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      image_path TEXT,
                      prediction TEXT,
                      datetime TEXT)''')
        conn.commit()
        conn.close()
init_db()

# Preprocess for FER2013 model (48x48 grayscale)
def preprocess_img(file):
    img = Image.open(file).convert('L').resize((48,48))
    img_arr = np.array(img) / 255.0
    img_arr = np.reshape(img_arr, (48,48,1))
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        img_file = request.files['img']
        img_path = os.path.join("static", img_file.filename)
        img_file.save(img_path)
        img_arr = preprocess_img(img_path)
        result = model.predict(img_arr)
        emotion = str(np.argmax(result))

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO users (name, image_path, prediction, datetime) VALUES (?, ?, ?, datetime('now'))",
                  (name, img_path, emotion))
        conn.commit()
        conn.close()

        return render_template('index.html', emotion=emotion, img_path=img_path)
    return render_template('index.html', emotion=None, img_path=None)

@app.route('/download')
def download_model():
    return send_file(MODEL_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
