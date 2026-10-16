# AI Image Classification and Inference Service

An AI-powered image classification application built using Python, TensorFlow, MobileNetV2, and Streamlit.

## Overview

This project provides an interactive web application that classifies uploaded images using a pretrained MobileNetV2 deep-learning model.

The application displays the top three predicted categories, confidence scores, and inference time.

## Features

- Upload JPG, JPEG, and PNG images.
- Validate image format and file size.
- Preprocess and resize images.
- Classify images using MobileNetV2.
- Display the top three predictions with confidence scores.
- Measure inference time.
- Warn users about low-confidence predictions.
- Handle corrupted and unsupported images.

## Technology Stack

- Python
- TensorFlow and Keras
- MobileNetV2
- Streamlit
- Pillow
- NumPy
- Scikit-learn
- Matplotlib

## Model Information

- **Model:** MobileNetV2
- **Framework:** TensorFlow / Keras
- **Pretrained weights:** ImageNet
- **Input image size:** 224 × 224 pixels
- **Output:** Top three predicted categories with confidence scores

The application uses a pretrained model. The model was not trained from scratch for this project.

The model recognises general ImageNet object categories and is not specifically trained to diagnose plant diseases.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bangarambavinagaraju-blip/ai-image-classification-service.git
```

### 2. Open the project directory

```bash
cd ai-image-classification-service
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Dataset and Evaluation

The pretrained model uses ImageNet weights.

For project evaluation, the model was tested on a subset of 100 images from the ImageNette validation dataset.

- **Evaluation samples:** 100 images
- **Top-1 accuracy:** 83%

This accuracy represents the evaluated subset only. It should not be interpreted as the model's overall accuracy across all ImageNet images.

## Error Handling and Reliability

The application includes image format validation, file-size checks, image preprocessing, and handling for corrupted or unsupported images.

A warning is displayed for low-confidence predictions. However, the model may still produce predictions for blank, unclear, or unfamiliar images.

## Live Application

[Open the deployed application](https://ai-image-classification-service-c3d5bedhhglubewxqe222l.streamlit.app/)

## GitHub Repository

[View the source code](https://github.com/bangarambavinagaraju-blip/ai-image-classification-service)

## Future Improvements

- Improve blank-image detection.
- Explore fine-tuning MobileNetV2.
- Evaluate performance on a larger test dataset.
- Optimise inference performance.

## Author

Nagaraju Bangarambavi
