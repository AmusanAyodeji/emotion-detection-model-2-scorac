import os
import sqlite3
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("model.h5")  # Load your trained model

DB_FILE = "emotion_users.db"

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

def preprocess_img(file):
    img = Image.open(file).convert('L').resize((28,28))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

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

port = int(os.environ.get("PORT", 10000))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port, debug=True)
