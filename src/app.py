import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil
import cv2
from group_predict import detect_and_classify_faces
from single_predict import predict_gender
from datetime import datetime

# Set page config
st.set_page_config(page_title="Gender Classification", page_icon="👤", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-text {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("👤 Gender Classification App")
st.markdown(
    """
This app uses deep learning to classify gender in both individual portraits and group photos.
Upload an image to get started!
"""
)

# Create tabs for single and group photo analysis
tab1, tab2 = st.tabs(["Individual Portrait", "Group Photo"])

with tab1:
    st.header("Individual Portrait Analysis")
    st.markdown("Upload a clear portrait photo of a single person.")

    single_image = st.file_uploader(
        "Choose a portrait photo", type=["jpg", "jpeg", "png"], key="single"
    )

    if single_image is not None:
        try:
            # Display the uploaded image
            image = Image.open(single_image)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Portrait", use_column_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict_gender(image)

                    # Display results with styling
                    st.markdown(
                        f"<p class='result-text'>Predicted Gender: {prediction}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p class='confidence-text'>Confidence: {confidence:.1f}%</p>",
                        unsafe_allow_html=True,
                    )

                    # Add a progress bar for confidence
                    st.progress(confidence / 100)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Group Photo Analysis")
    st.markdown(
        "Upload a group photo to detect and classify gender for multiple people."
    )

    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0,
        max_value=100,
        value=50,
        help="Adjust this to filter out low-confidence predictions",
    )

    group_image = st.file_uploader(
        "Choose a group photo", type=["jpg", "jpeg", "png"], key="group"
    )

    if group_image is not None:
        try:
            # Save the uploaded file temporarily
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(temp_path, "wb") as f:
                f.write(group_image.getbuffer())

            # Load model and process image
            with st.spinner("Analyzing group photo..."):
                model = load_model()
                results = detect_and_classify_faces(
                    temp_path, model, min_confidence=confidence_threshold
                )

            # Remove temporary file
            os.remove(temp_path)

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                # Display annotated image
                st.image(
                    results["annotated_image"],
                    caption="Analyzed Group Photo",
                    use_column_width=True,
                )

            with col2:
                # Display statistics
                st.markdown("### Analysis Results")

                # Total faces
                st.metric("Total Faces Detected", results["total_faces"])

                # Gender distribution
                st.markdown("#### Gender Distribution")
                if results["total_faces"] > 0:
                    # Calculate percentages
                    male_percent = (results["males"] / results["total_faces"]) * 100
                    female_percent = (results["females"] / results["total_faces"]) * 100

                    # Create a pie chart
                    fig, ax = plt.subplots(figsize=(6, 6))
                    colors = [
                        "#FF7F50",
                        "#87CEEB",
                    ]  # Coral for males, Sky Blue for females
                    ax.pie(
                        [results["males"], results["females"]],
                        labels=["Males", "Females"],
                        autopct="%1.1f%%",
                        colors=colors,
                        startangle=90,
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                    # Display counts
                    st.markdown(
                        f"""
                    - Males: {results['males']} ({male_percent:.1f}%)
                    - Females: {results['females']} ({female_percent:.1f}%)
                    """
                    )
                else:
                    st.info("No faces detected in the image.")

                # Display average confidence
                if results["results"]:
                    avg_confidence = sum(
                        r["confidence"] for r in results["results"]
                    ) / len(results["results"])
                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")

        except Exception as e:
            st.error(f"Error processing group photo: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Model class
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    model.load_state_dict(
        torch.load("best_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


# Image transformation
def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


# Prediction function
def predict(model, image):
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        return probability


def process_batch_images(model, uploaded_files):
    results = []

    for file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as tmp_file:
                # Write the uploaded file to the temporary file
                shutil.copyfileobj(file, tmp_file)

            # Process the image
            image = Image.open(tmp_file.name).convert("RGB")
            transformed_image = transform_image(image)
            probability = predict(model, transformed_image)

            gender = "Male" if probability > 0.5 else "Female"
            confidence = probability if probability > 0.5 else 1 - probability

            results.append(
                {
                    "filename": file.name,
                    "predicted_gender": gender,
                    "confidence": confidence * 100,
                }
            )

            # Clean up the temporary file
            os.unlink(tmp_file.name)

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    return pd.DataFrame(results)


# Main app
def main():
    st.title("Gender Classification Model")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Model Details",
            "Test Model",
            "Group Photo Analysis",
            "Batch Processing",
            "Performance Metrics",
        ],
    )

    if page == "Home":
        st.header("Welcome to Gender Classification App")
        st.write(
            """
        This application uses a deep learning model (ResNet18) to classify gender from facial images.
        The model has been trained on a large dataset and achieves 98.22% accuracy in gender prediction.
        
        ### Features:
        - Upload and test individual images
        - Analyze group photos with multiple faces
        - Batch process multiple images
        - View detailed model architecture and training process
        - Examine model performance metrics
        - Real-time predictions with confidence scores
        """
        )

        # Display sample images if available
        if os.path.exists("Dataset/Test"):
            st.subheader("Sample Images from Test Set")
            test_path = Path("Dataset/Test")
            sample_images = list(test_path.glob("*/*"))[:4]
            cols = st.columns(4)
            for col, img_path in zip(cols, sample_images):
                with col:
                    st.image(str(img_path), caption=img_path.parent.name)

    elif page == "Model Details":
        st.header("Model Architecture and Training Details")

        # Load and display technical details
        with open("technical_details.md", "r") as file:
            st.markdown(file.read())

    elif page == "Test Model":
        st.header("Test Single Image")

        # Load model
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                # Display original image
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)

                # Make prediction
                transformed_image = transform_image(image)
                probability = predict(model, transformed_image)

                # Display results
                st.subheader("Prediction Results:")
                col1, col2 = st.columns(2)

                with col1:
                    gender = "Male" if probability > 0.5 else "Female"
                    confidence = probability if probability > 0.5 else 1 - probability
                    st.write(f"Predicted Gender: **{gender}**")
                    st.write(f"Confidence: **{confidence*100:.2f}%**")

                with col2:
                    # Create confidence visualization
                    fig, ax = plt.subplots(figsize=(8, 1))
                    ax.barh(["Confidence"], [confidence * 100], color="blue")
                    ax.set_xlim(0, 100)
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif page == "Group Photo Analysis":
        st.header("Analyze Group Photo")

        # Load model
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a group photo...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)

                # Process group photo
                results = detect_and_classify_faces(tmp_file.name, model)

                # Clean up
                os.unlink(tmp_file.name)

                # Display results
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.image(
                        results["annotated_image"],
                        caption="Analyzed Image",
                        use_column_width=True,
                    )

                with col2:
                    st.subheader("Analysis Results")
                    st.write(f"Total faces detected: **{results['total_faces']}**")
                    st.write(f"Males detected: **{results['males']}**")
                    st.write(f"Females detected: **{results['females']}**")

                    # Create pie chart
                    if results["total_faces"] > 0:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        plt.pie(
                            [results["males"], results["females"]],
                            labels=["Males", "Females"],
                            autopct="%1.1f%%",
                            colors=["blue", "red"],
                        )
                        plt.title("Gender Distribution")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing group photo: {str(e)}")

    elif page == "Batch Processing":
        st.header("Batch Process Multiple Images")

        # Load model
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Processing images..."):
                # Process images
                results_df = process_batch_images(model, uploaded_files)

                # Display results
                st.subheader("Results:")
                st.dataframe(results_df)

                # Create visualizations
                col1, col2 = st.columns(2)

                with col1:
                    # Gender distribution
                    st.subheader("Gender Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=results_df, x="predicted_gender")
                    plt.title("Predicted Gender Distribution")
                    st.pyplot(fig)

                with col2:
                    # Confidence distribution
                    st.subheader("Confidence Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(data=results_df, x="confidence", bins=20)
                    plt.title("Prediction Confidence Distribution")
                    st.pyplot(fig)

                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

    elif page == "Performance Metrics":
        st.header("Model Performance Metrics")

        # Display test results
        st.subheader("Test Set Results")
        st.code(
            """
Test Accuracy: 98.22%

Classification Report:
              precision    recall  f1-score   support

      Female       0.98      0.99      0.98      2851
        Male       0.98      0.98      0.98      2149

    accuracy                           0.98      5000
   macro avg       0.98      0.98      0.98      5000
weighted avg       0.98      0.98      0.98      5000
        """
        )

        # Display training metrics
        if os.path.exists("training_metrics.png"):
            st.subheader("Training Progress")
            st.image("training_metrics.png", caption="Training and Validation Metrics")

        # Display ROC curve
        if os.path.exists("roc_curve.png"):
            st.subheader("ROC Curve")
            st.image("roc_curve.png", caption="Receiver Operating Characteristic Curve")

        # Display confusion matrix
        if os.path.exists("confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image("confusion_matrix.png", caption="Test Set Confusion Matrix")


if __name__ == "__main__":
    main()
