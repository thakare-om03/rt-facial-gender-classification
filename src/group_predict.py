import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from src.utils import detect_faces, align_face

def process_group_photo(image_path, model_path='models/best_model.h5'):
    """
    Process a group photo to detect and classify genders of individuals.
    Utilizes MTCNN for face detection with a Haar Cascade fallback,
    aligns faces and annotates them with gender and confidence scores.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    orig_image = image.copy()

    # Attempt face detection using the provided utility function.
    faces = detect_faces(image)
    if len(faces) == 0:
        # Fallback to Haar Cascade if MTCNN does not detect any faces.
        print("No faces detected with MTCNN, using Haar Cascade fallback.")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = []
        for (x, y, w, h) in haar_faces:
            # Approximate key-points based on the bounding box.
            keypoints = {"left_eye": (x + int(w * 0.3), y + int(h * 0.4)),
                         "right_eye": (x + int(w * 0.7), y + int(h * 0.4)),
                         "nose": (x + w // 2, y + h // 2),
                         "mouth_left": (x + int(w * 0.35), y + int(h * 0.75)),
                         "mouth_right": (x + int(w * 0.65), y + int(h * 0.75))}
            faces.append(((x, y, w, h), keypoints))

    model = load_model(model_path)
    results = []

    for (box, keypoints) in faces:
        x, y, w, h = box
        face = image[y:y + h, x:x + w]
        # Align the face using the detected keypoints.
        aligned_face = align_face(face, keypoints)
        # Resize the aligned face to the model's input shape.
        face_input = cv2.resize(aligned_face, (224, 224))
        face_input = face_input.astype('float32') / 255.0
        face_input = np.expand_dims(face_input, axis=0)
        # Predict gender and compute a confidence score.
        pred = model.predict(face_input)
        prob = pred[0]
        confidence = np.max(prob)
        gender = 'Male' if np.argmax(prob) == 0 else 'Female'
        results.append(((x, y, w, h), gender, confidence * 100))
        cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_image, f"{gender} {confidence*100:.1f}%", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_image_rgb)
    plt.axis('off')
    plt.title("Group Photo: Detected Faces and Gender Labels")
    plt.show()

    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'dataset/Test/group_photo.jpg'
    
    results = process_group_photo(image_path)
    for bbox, gender, confidence in results:
        print(f"Face at {bbox} predicted as {gender} with {confidence:.1f}% confidence")
