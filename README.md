This project uses deep learning to automatically classify ceramic tiles as good or defective, helping automate quality inspection in manufacturing environments. A Convolutional Neural Network (CNN) was trained on a custom image dataset, and a FastAPI service was built to provide real-time predictions through an image upload interface.

Key Features:
Custom dataset with “good” and “defect” tile classes
CNN-based image classification using TensorFlow/Keras
Saved trained model for reuse and deployment
FastAPI backend for real-time inference
Swagger UI for easy testing of image predictions

Technologies Used:
Python, TensorFlow, Keras, FastAPI, Uvicorn, Pillow, NumPy
