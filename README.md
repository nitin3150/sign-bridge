# Sign Bridge – Real-Time Gesture Recognition Benchmark

An AI-powered benchmarking system for evaluating hand gesture recognition models in real-time using sign language datasets.

## 🚀 Features
- Benchmarks hand gesture models using MediaPipe and OpenCV.
- Evaluates gesture recognition accuracy and response time.
- Integrated with gTTS for real-time feedback testing.

## 📊 Benchmark Goals
- Test multiple ML pipelines for gesture translation speed and accuracy.
- Compare frame rates, latency, and gesture recognition precision.
- Automate performance reporting with video and metrics.

## 🛠️ Tech Stack
- Python, OpenCV, MediaPipe
- TensorFlow/Keras
- gTTS (Google Text-to-Speech)

## 📁 Structure
- `/models`: Pretrained and custom models
- `/scripts`: Real-time inference and evaluation
- `/data`: Sample gesture datasets
- `/output`: Logs and performance metrics

## 📌 Reproducibility
Install dependencies using `pip install -r requirements.txt`.  
Run `python scripts/main_pipeline.py` to start evaluation.

## 📈 Sample Output
- Recognition Accuracy: 94%
- Real-time translation at 30 FPS
