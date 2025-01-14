# app/train.py
import tensorflow as tf
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SignLanguageTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
    def process_image(self, image_path):
        """Process a single image and extract hand landmarks."""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks
        return None

    def prepare_dataset(self, data_dir):
        """Prepare dataset from directory of images."""
        images = []
        labels = []
        
        # Assuming data_dir contains folders named A-Z with corresponding images
        for letter in range(ord('A'), ord('Z')+1):
            letter_dir = f"{data_dir}/{chr(letter)}"
            for img_path in Path(letter_dir).glob("*.jpg"):
                landmarks = self.process_image(str(img_path))
                if landmarks:
                    images.append(landmarks)
                    labels.append(chr(letter))
        
        return np.array(images), np.array(labels)

    def train_model(self, data_dir, epochs=50, batch_size=32):
        """Train the sign language model."""
        # Prepare data
        X, y = self.prepare_dataset(data_dir)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Reshape data for the model
        X_train = X_train.reshape(-1, 21, 3)
        X_test = X_test.reshape(-1, 21, 3)
        
        # Create and compile model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(21, 3)),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Save model
        model.save('sign_language_model.h5')
        
        return model, history, label_encoder

# Example usage script
if __name__ == "__main__":
    trainer = SignLanguageTrainer()
    
    # Specify your dataset directory
    DATA_DIR = "path/to/your/dataset"
    
    # Train the model
    model, history, label_encoder = trainer.train_model(DATA_DIR)
    
    # Save label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
