import cv2
import numpy as np
from deepface import DeepFace
import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Model paths (UPDATE THESE PATHS)
MODEL_DIR = "models"
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt.txt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def download_models():
    """Download required face detection models if missing"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Prototxt file
    if not os.path.exists(PROTOTXT_PATH):
        print("Downloading deploy.prototxt.txt...")
        os.system(f"curl https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -o {PROTOTXT_PATH}")
    
    # Caffe model
    if not os.path.exists(CAFFEMODEL_PATH):
        print("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
        os.system(f"curl https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -o {CAFFEMODEL_PATH} -L")

# Check and download models
download_models()

# Verify model files exist
if not all([os.path.exists(PROTOTXT_PATH), os.path.exists(CAFFEMODEL_PATH)]):
    raise FileNotFoundError("Missing model files. Please check the models directory.")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load face detection model
face_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Boundary checks
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w-1, endX), min(h-1, endY)
            
            face = frame[startY:endY, startX:endX]
            
            if face.size == 0:
                continue
                
            try:
                analysis = DeepFace.analyze(face, actions=['gender'], 
                                          enforce_detection=False, silent=True)
                gender = analysis[0]['dominant_gender']
                confidence = analysis[0]['gender'][gender]
                
                label = f"{gender} {confidence:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
                cv2.putText(frame, label, (startX, startY-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            except Exception as e:
                print(f"Analysis error: {str(e)}")
                continue

    cv2.imshow('Gender Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
