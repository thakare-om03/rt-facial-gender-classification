import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter


def load_model(model_path="best_model.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

        # Adjust the threshold to be more balanced
        # This helps prevent misclassifying females as males
        threshold = 0.6  # Increased threshold for male classification
        prediction = "Male" if probability > threshold else "Female"
        confidence = probability if probability > threshold else 1 - probability
        return prediction, confidence * 100


def preprocess_face(face):
    # Convert to grayscale for histogram equalization
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Convert back to BGR
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Blend with original image
    result = cv2.addWeighted(face, 0.7, enhanced, 0.3, 0)

    return result


def align_face(image, face_rect, landmarks):
    if landmarks is None or len(landmarks) < 2:
        return image

    # Get eye centers
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Calculate angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the center of the face
    center = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned


def detect_and_classify_faces(image_path, model, min_confidence=50):
    # Load cascade classifiers
    face_cascade_front = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade_profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Resize if image is too large
    max_dimension = 1200
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect faces with different scale factors
    faces_front = face_cascade_front.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces_profile = face_cascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Combine detections
    faces = list(faces_front) + list(faces_profile)

    # Remove overlapping detections
    if len(faces) > 0:
        faces = np.array(faces)
        pick = non_max_suppression(faces, 0.3)
        faces = faces[pick]

    results = []
    img_with_boxes = img.copy()

    for x, y, w, h in faces:
        # Verify face by checking for eyes
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Only process if eyes are detected
        if len(eyes) >= 1:
            # Extract face with margin
            margin = int(0.3 * w)  # Increased margin for better context
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, img.shape[1])
            y2 = min(y + h + margin, img.shape[0])

            face = img[y1:y2, x1:x2]

            # Preprocess the face
            face = preprocess_face(face)

            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Transform and predict
            face_tensor = transform_image(face_pil)
            prediction, confidence = predict(model, face_tensor)

            if confidence >= min_confidence:
                results.append(
                    {
                        "gender": prediction,
                        "confidence": confidence,
                        "bbox": (x, y, w, h),
                    }
                )

                # Draw rectangle and label
                color = (
                    (0, 128, 255) if prediction == "Male" else (255, 128, 0)
                )  # Orange for male, Blue for female
                thickness = 2

                # Draw filled rectangle for label background
                label = f"{prediction} ({confidence:.1f}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                cv2.rectangle(
                    img_with_boxes, (x, y - 30), (x + label_size[0], y), color, -1
                )

                # Draw face rectangle
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, thickness)

                # Add label text
                cv2.putText(
                    img_with_boxes,
                    label,
                    (x, y - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )

    # Convert summary statistics
    total_faces = len(results)
    males = sum(1 for r in results if r["gender"] == "Male")
    females = sum(1 for r in results if r["gender"] == "Female")

    summary = {
        "total_faces": total_faces,
        "males": males,
        "females": females,
        "results": results,
        "annotated_image": cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB),
    }

    return summary


def non_max_suppression(boxes, overlap_thresh):
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []

    # Convert to float
    boxes = boxes.astype("float")
    pick = []

    # Coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute area of bounding boxes and sort by bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height of bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return pick


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and classify gender in group photos"
    )
    parser.add_argument("image_path", type=str, help="Path to the group photo")
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to the model file"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=50,
        help="Minimum confidence threshold (0-100)",
    )
    parser.add_argument("--output", type=str, help="Path to save annotated image")
    args = parser.parse_args()

    try:
        # Load model
        print("Loading model...")
        model = load_model(args.model)

        # Process image
        print("Processing image...")
        results = detect_and_classify_faces(args.image_path, model, args.min_confidence)

        # Print results
        print(f"\nResults:")
        print(f"Total faces detected: {results['total_faces']}")
        print(f"Males detected: {results['males']}")
        print(f"Females detected: {results['females']}")

        # Save annotated image if output path is provided
        if args.output:
            plt.figure(figsize=(12, 8))
            plt.imshow(results["annotated_image"])
            plt.axis("off")
            plt.savefig(args.output, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"\nAnnotated image saved to: {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
