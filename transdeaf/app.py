from flask import Flask, render_template, request, jsonify
import cv2
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import base64
import io
from PIL import Image

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Global variables to store model and labels
model = None
labels = []

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/collect_data', methods=['GET', 'POST'])
def collect_data_page():
    return render_template('collect_data.html')


@app.route('/collect_frame', methods=['POST'])
def collect_frame():
    data = request.get_json()
    label = data['label']
    count = data['count']
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)

    save_path = 'data'
    label_path = os.path.join(save_path, label)
    os.makedirs(label_path, exist_ok=True)

    frame_path = os.path.join(label_path, f'{label}_{count}.jpg')
    cv2.imwrite(frame_path, frame)
    return jsonify({"status": "Frame saved"})

@app.route('/load_data', methods=['GET'])
def api_load_data():
    global labels
    X_train, y_train, labels = load_data()
    if X_train.size == 0 or y_train.size == 0:
        return jsonify({"status": "No data loaded. Please check the data directory and preprocessing steps."}), 400
    y_train = to_categorical(y_train)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    return jsonify({"status": "Data loaded successfully", "labels": labels})

@app.route('/train_model', methods=['GET', 'POST'])
def train_model_page():
    if request.method == 'POST':
        global model, labels
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        num_classes = y_train.shape[1]
        model,history = train_model(X_train, y_train, num_classes)
        model.save('gesture_model.h5')
        return jsonify({"accuracy":history.history['accuracy'],"status": "Model trained successfully"})

    return render_template('train_model.html')

@app.route('/inference', methods=['GET'])
def inference_page():
    return render_template('inference.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global model, labels
    if model is None:
        model = tf.keras.models.load_model('gesture_model.h5')

    # Decode image from base64
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)

    # Preprocess frame and make prediction
    landmarks = preprocess_frame(frame)
    if landmarks:
        landmarks = np.array(landmarks).reshape(1, 21, 3)
        prediction = model.predict(landmarks)
        predicted_label = np.argmax(prediction)
        return jsonify({"label": labels[predicted_label]})
    return jsonify({"label": "No hand detected"})

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            return landmarks
    return None

def load_data(data_path='data'):
    X, y = [], []
    labels = os.listdir(data_path)
    for label in labels:
        label_path = os.path.join(data_path, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            frame = cv2.imread(img_path)
            landmarks = preprocess_frame(frame)
            if landmarks:
                X.append(landmarks)
                y.append(labels.index(label))
    return np.array(X), np.array(y), labels

def train_model(X_train, y_train, num_classes):
    model = Sequential([
        Flatten(input_shape=(21, 3)),  # 21 landmarks with 3 coordinates (x, y, z)
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    
    return model,history


if __name__ == '__main__':
    app.run(debug=True)