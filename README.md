# AI-Based Defect Detection for Manufacturing

 # project Overview
 AI-Based Defect Detection for Manufacturing is a deep learning project designed to automatically identify defects in ceramic tiles using image classification.
Using a Convolutional Neural Network (CNN), the system classifies tile images as either good or defective.
The project includes dataset creation, preprocessing, model training, evaluation, and a FastAPI service for real-time defect detection.

The goal is to support manufacturing environments by providing automated, accurate, and fast quality inspection.

# Key Features
Automatic dataset generation with synthetic tile images

Image preprocessing, normalization, and structured dataset splitting

Deep learning model built using TensorFlow/Keras

Local prediction script for testing individual images

FastAPI endpoint for real-time defect detection

Supports easy image uploads and JSON output

# Technologies Used
Python

TensorFlow / Keras

FastAPI

Uvicorn

Pillow

NumPy

# Getting Started
# Prerequisites

Install required packages:
pip install tensorflow fastapi uvicorn pillow numpy python-multipart
#
# Running the Project
Generate the dataset:python create_small_dataset.py
#
# Train the model:
train_model:python train_model.py
#
# Start the API server:
tart the API server:uvicorn api_app:app --reload
#
# Open the API documentation:
click the URL:http://127.0.0.1:8000/docs
#
# Example Output
Example Output:
{
  "probability_good": 0.82,
  "predicted_class": "good"
}
#
# Acknowledgments

TensorFlow/Keras open-source developers

FastAPI for providing a high-performance API framework

Python’s scientific stack supporting deep learning workflows










