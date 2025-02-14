import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.utils import detect_faces, align_face

def predict_single(image_path, model_path='models/best_model.h5'):
    """
    Predict gender from the first detected face in an image.
    Annotates and displays the image with the prediction.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None

    faces = detect_faces(image)
    if len(faces) == 0:
        print("No faces detected.")
        return None

    (box, keypoints) = faces[0]
    x, y, w, h = box
    face = image[y:y+h, x:x+w]
    aligned_face = align_face(face, keypoints)
    face_input = cv2.resize(aligned_face, (224, 224))
    face_input = face_input.astype('float32') / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    model = load_model(model_path)
    pred = model.predict(face_input)
    gender = 'Male' if np.argmax(pred) == 0 else 'Female'

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Predicted Gender: {gender}")
    plt.show()
    return gender

def predict_group(image_path, model_path='models/best_model.h5'):
    """
    Predict gender for each detected face in a group photo.
    Annotates the image with bounding boxes and gender labels for all faces.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None

    orig_image = image.copy()
    faces = detect_faces(image)
    if len(faces) == 0:
        print("No faces detected.")
        return None

    model = load_model(model_path)
    results = []

    for (box, keypoints) in faces:
        x, y, w, h = box
        face = image[y:y+h, x:x+w]
        aligned_face = align_face(face, keypoints)
        face_input = cv2.resize(aligned_face, (224, 224))
        face_input = face_input.astype('float32') / 255.0
        face_input = np.expand_dims(face_input, axis=0)
        pred = model.predict(face_input)
        gender = 'Male' if np.argmax(pred) == 0 else 'Female'
        results.append(((x, y, w, h), gender))
        cv2.rectangle(orig_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(orig_image, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_image_rgb)
    plt.axis('off')
    plt.title("Group Photo: Detected Faces and Gender Labels")
    plt.show()
    return results

if __name__ == '__main__':
    import sys
    # Usage: python predict.py [single/group] [image_path]
    if len(sys.argv) < 3:
        print("Usage: python predict.py [single/group] [image_path]")
        sys.exit(1)
    mode = sys.argv[1]
    image_path = sys.argv[2]
    if mode == 'single':
        gender = predict_single(image_path)
        print("Predicted Gender:", gender)
    elif mode == 'group':
        res = predict_group(image_path)
        if res:
            for bbox, gender in res:
                print(f"Face at {bbox} predicted as {gender}")
    else:
        print("Invalid mode. Use 'single' or 'group'.")