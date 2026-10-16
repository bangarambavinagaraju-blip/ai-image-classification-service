# AI Image Classification and Inference Service

An AI-powered image classification application built with Python,
TensorFlow, MobileNetV2, and Streamlit.

## Overview

This project provides an interactive web application that classifies
uploaded images using a pretrained MobileNetV2 deep learning model.

The application displays the top three predicted categories, confidence
scores, and inference time.

## Features

- Image upload for JPG, JPEG, and PNG files
- Image format and file-size validation
- Image preprocessing and resizing
- MobileNetV2-based image classification
- Top-three predictions with confidence scores
- Inference time measurement
- Warning for low-confidence predictions
- Error handling for corrupted or unsupported images
- Interactive Streamlit interface

## Technology Stack

- Python
- TensorFlow
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Scikit-learn
- Matplotlib

## Model Information

The application uses MobileNetV2 pretrained on ImageNet.

The model recognizes general object categories from ImageNet.
It is not specifically trained for plant disease diagnosis.

## Installation

Clone the repository:

```bash
git clone https://github.com/bangarambavinagaraju-blip/ai-image-classification-service.git

## How to Run the Application

After installing the required dependencies, run the following command:

```bash
streamlit run app.py
