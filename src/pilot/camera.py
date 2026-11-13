import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Tell PyAutoGUI not to crash if cursor touches screen edges
pyautogui.FAILSAFE = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Key landmark indices in MediaPipe:
LEFT_IRIS = 468
RIGHT_IRIS = 473

# Left eye outline: outer, inner, top, bottom
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

def get_eye_metrics(frame, face_mesh):
    """
    Processes a frame and returns:
    - iris positions
    - ear (Eye Aspect Ratio) to detect blinks
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
        
    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. Iris center coordinates
    l_iris_x = landmarks[LEFT_IRIS].x * w
    l_iris_y = landmarks[LEFT_IRIS].y * h
    
    # 2. Eye Aspect Ratio (EAR) for blink detection
    # Height between top and bottom eyelid
    top = np.array([landmarks[LEFT_EYE_TOP].x * w, landmarks[LEFT_EYE_TOP].y * h])
    bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * w, landmarks[LEFT_EYE_BOTTOM].y * h])
    # Width between inner and outer corners
    inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    
    ear = np.linalg.norm(top - bottom) / np.linalg.norm(inner - outer)
    
    return {
        "iris": (l_iris_x, l_iris_y),
        "ear": ear,
        "is_blinking": ear < 0.20  # Under 0.20 typically means eye is closed
    }



def startup(): #this will run after the keyword, will pop up the gui and start the camera, then call the calibration stuff. then this will  create the 3d map out of the eye vectors so we need to do eye vector analysis as well as distnace metrix from the camera.
    print("Starting Camera and face landmarker")
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh:

        calib_data = calibrate(cap,face_mesh)

        if calib_data:
            watcheyes(cap,face_mesh,calib_data)
        


def calibrate():

def watcheyes(): #this is the normal running state of the application