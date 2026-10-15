
# ============================================================
# AI IMAGE CLASSIFICATION AND INFERENCE SERVICE
# Streamlit + MobileNetV2
# ============================================================

import io
import time

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# ------------------------------------------------------------
# 1. Configure the Streamlit application
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Vision",
    page_icon="🧠",
    layout="wide",
)

st.title("AI Vision - Image Classification")
st.write(
    "Upload an image to classify objects using a pretrained "
    "MobileNetV2 deep learning model."
)

st.info(
    "This model recognizes general ImageNet object categories. "
    "It is not specifically trained to diagnose plant diseases."
)

# ------------------------------------------------------------
# 2. Load the pretrained model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return MobileNetV2(
        weights="imagenet",
        include_top=True,
    )

# ------------------------------------------------------------
# 3. Validate the uploaded image
# ------------------------------------------------------------

def validate_image(image_bytes, filename):

    if not image_bytes:
        return False, "The uploaded file is empty."

    if len(image_bytes) > 5 * 1024 * 1024:
        return False, "Maximum allowed image size is 5 MB."

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return False, "Upload a JPG, JPEG, or PNG image."

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:

            if image.format not in ("JPEG", "PNG"):
                return False, "The actual file format is unsupported."

            width, height = image.size

            if width <= 0 or height <= 0:
                return False, "Invalid image dimensions."

            if width * height > 40_000_000:
                return False, "Image dimensions are too large."

            image.verify()

        return True, "Image validation successful."

    except (UnidentifiedImageError, OSError, ValueError):
        return False, "The image is corrupted or cannot be opened."

# ------------------------------------------------------------
# 4. Preprocess the image
# ------------------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return preprocess_input(image_array)

# ------------------------------------------------------------
# 5. Perform image classification
# ------------------------------------------------------------

def classify_image(image, model):

    processed_image = preprocess_image(image)

    start_time = time.perf_counter()

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    inference_time = time.perf_counter() - start_time

    decoded = decode_predictions(
        predictions,
        top=3,
    )[0]

    results = []

    for class_id, class_name, confidence in decoded:

        results.append({
            "class_id": class_id,
            "class_name": class_name.replace("_", " ").title(),
            "confidence": float(confidence) * 100,
        })

    return results, inference_time

# ------------------------------------------------------------
# 6. Image upload interface
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Maximum file size: 5 MB.",
)

# ------------------------------------------------------------
# 7. Process the image and display results
# ------------------------------------------------------------

if uploaded_file is None:

    st.info("Choose an image to start classification.")

else:

    image_bytes = uploaded_file.getvalue()

    valid, message = validate_image(
        image_bytes,
        uploaded_file.name,
    )

    if not valid:

        st.error(message)

    else:

        st.success(message)

        try:

            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")

            image_column, result_column = st.columns(2)

            with image_column:
                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True,
                )

            with st.spinner("Loading model and analyzing image..."):

                model = load_model()

                results, inference_time = classify_image(
                    image,
                    model,
                )

            with result_column:

                st.subheader("Classification Results")

                for rank, result in enumerate(results, start=1):

                    st.write(
                        f"**{rank}. {result['class_name']}**"
                    )

                    st.progress(
                        min(
                            max(result["confidence"] / 100, 0.0),
                            1.0,
                        )
                    )

                    st.write(
                        f"Confidence: {result['confidence']:.2f}%"
                    )

                if results[0]["confidence"] < 60:

                    st.warning(
                        "The top prediction has less than 60% confidence. "
                        "The result may be uncertain."
                    )

            st.divider()

            metric_one, metric_two = st.columns(2)

            metric_one.metric(
                "Inference Time",
                f"{inference_time:.3f} seconds",
            )

            metric_two.metric(
                "Predictions Displayed",
                len(results),
            )

            st.caption(
                "Confidence scores are model outputs and do not "
                "guarantee that a prediction is correct."
            )

        except Exception as error:

            st.error(
                "An error occurred while processing the image."
            )

            st.caption(
                f"Error details: {type(error).__name__}: {error}"
            )
