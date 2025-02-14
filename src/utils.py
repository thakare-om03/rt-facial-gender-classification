from mtcnn import MTCNN
import cv2
import numpy as np

def detect_faces(image):
    """
    Detect faces in the image using MTCNN.
    Returns a list of tuples: ((x, y, w, h), keypoints)
    """
    detector = MTCNN()
    results = detector.detect_faces(image)
    faces = []
    for r in results:
        x, y, w, h = r['box']
        keypoints = r['keypoints']
        faces.append(((x, y, w, h), keypoints))
    return faces

def align_face(face, keypoints):
    """
    Align the face based on detected keypoints.
    This simple placeholder returns the face as is.
    For better accuracy, an affine transformation can be applied.
    """
    return face

def random_occlusion(image):
    """
    Randomly occlude a small region of the image to simulate occlusion.
    This is used as a preprocessing function for data augmentation.
    """
    h, w, _ = image.shape
    occ_w, occ_h = int(w * 0.2), int(h * 0.2)
    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)
    image[y:y+occ_h, x:x+occ_w, :] = 0
    return image
