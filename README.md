# ISL_Gesture_Recognition
A real-time Indian Sign Language (ISL) gesture recognition system developed as part of my NTCC (Non-Teaching Credit Course) project.

# Project Overview
This project is focused on recognizing static hand gestures of ISL alphabets using a Convolutional Neural Network (CNN). It performs real-time gesture recognition using a webcam and was built individually as part of my college coursework.

# Key Features
🔤 Real-time ISL alphabet gesture recognition

📷 Webcam-based live prediction

🧠 Custom-trained CNN model

📚 Academic project under NTCC submission

# Tech Stack
Language: Python

Libraries: TensorFlow / Keras, OpenCV, NumPy

Model: CNN (Convolutional Neural Network)

# Workflow
Data Collection

Static gesture images of ISL alphabets were collected and labeled.

Model Training

A CNN was trained on the dataset to recognize hand gestures.

Real-Time Recognition

OpenCV was used to access webcam input and classify gestures in real time.

# Getting Started
Install Dependencies
pip install tensorflow opencv-python numpy

Run the Real-Time Recognition
python test.py

# Project Structure

isl-gesture-recognition/
│
├── dataset/               # ISL gesture image dataset
├── model/                 # Trained model files (e.g., model.h5)
├── train.py               # Script to train the CNN
├── test.py                # Real-time gesture recognition using webcam
└── README.md
# Model Results
Accuracy Achieved: ~95% 

Model Architecture: CNN with 3 convolutional layers 

# Author
Mayank Singh
B.Tech CSE, Amity University Uttar Pradesh
Email: thisismayank128@gmail.com

# Acknowledgment
This project was submitted as part of the NTCC (Non-Teaching Credit Course) requirement for my undergraduate program.
