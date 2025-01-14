# Project Structure
"""
sign_bridge/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── camera.py
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
"""

# requirements.txt
"""
fastapi==0.68.1
uvicorn==0.15.0
python-multipart==0.0.5
opencv-python==4.5.3.56
mediapipe==0.8.7.3
numpy==1.21.2
gTTS==2.2.3
tensorflow==2.6.0
"""

# app/main.py
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import base64
import json
from .model import SignLanguageModel
from .camera import CameraFeed

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize model
model = SignLanguageModel()
camera = CameraFeed()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            frame = camera.decode_frame(data)
            
            # Process frame
            results = hands.process(frame)
            
            if results.multi_hand_landmarks:
                # Extract hand landmarks
                landmarks = camera.extract_landmarks(results.multi_hand_landmarks[0])
                
                # Get prediction
                prediction = model.predict(landmarks)
                
                # Convert text to speech
                audio_file = f"temp_{prediction}.mp3"
                tts = gTTS(text=prediction, lang='en')
                tts.save(audio_file)
                
                # Send prediction and audio back to client
                with open(audio_file, "rb") as f:
                    audio_bytes = base64.b64encode(f.read()).decode()
                
                response = {
                    "prediction": prediction,
                    "audio": audio_bytes
                }
                
                await websocket.send_json(response)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

# app/model.py
import tensorflow as tf
import numpy as np

class SignLanguageModel:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(21, 3)),  # 21 landmarks, 3 coordinates each
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')  # 26 letters
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, landmarks):
        # Preprocess landmarks
        landmarks = np.array(landmarks).reshape(1, 21, 3)
        
        # Get prediction
        prediction = self.model.predict(landmarks)
        return chr(ord('A') + np.argmax(prediction))

# app/camera.py
import cv2
import numpy as np
import base64

class CameraFeed:
    def decode_frame(self, frame_data):
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        img_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return landmarks
