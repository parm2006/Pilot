import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg

pg.FAILSAFE = True
pg.PAUSE = 0
SCREEN_W,SCREEN_H = pg.size()

NOSE_TIP = 1
LEFT_IRIS = 468
L_EYE_INNER , L_EYE_OUTER = 133,33
L_EYE_TOP, L_EYE_BOTTOM = 159,145

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv2.VideoCapture(0)
tracking_active=False

smooth_x,smooth_y = SCREEN_W / 2, SCREEN_H / 2
anchor_x, anchor_y = smooth_x,smooth_y

DEADZONE_PX = 6 #anti jitter


ALPHA = 0.1

mouse_down = False
last_good_iris_y = 0.0
last_good_iris_x = 0.0
blink_counter = 0


while cap.isOpened():
    success,frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # 1. HEAD POSE: Normalize nose position within a central active box [0.35, 0.65]
        head_x = np.clip((lm[NOSE_TIP].x - 0.35) / 0.30, 0.0, 1.0)
        head_y = np.clip((lm[NOSE_TIP].y - 0.35) / 0.30, 0.0, 1.0)

        # 2. EYE GAZE: Relative position of left iris between inner and outer corners
        eye_width = abs(lm[L_EYE_INNER].x - lm[L_EYE_OUTER].x) + 1e-6
        eye_height = abs(lm[L_EYE_BOTTOM].y - lm[L_EYE_TOP].y) + 1e-6

        ear = eye_height/eye_width


        iris_rel_x = (lm[LEFT_IRIS].x - lm[L_EYE_OUTER].x) / eye_width - 0.5
        if ear < 0.3:  # Your natural open EAR is ~0.40
            iris_rel_x = last_good_iris_x
            iris_rel_y = last_good_iris_y
            blink_counter += 1

            if tracking_active and blink_counter == 5:
                print("CLICK FIRED!")
                pg.click()
        else:
            iris_rel_x = (lm[LEFT_IRIS].x - lm[L_EYE_OUTER].x) / eye_width - 0.5
            iris_rel_y = (lm[LEFT_IRIS].y - lm[L_EYE_TOP].y) / eye_height - 0.5
            last_good_iris_x = iris_rel_x
            last_good_iris_y = iris_rel_y
            blink_counter = 0



        cv2.putText(frame, f"EAR: {round(ear, 2)} | Blink: {blink_counter}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. FUSE HEAD + EYE (80% head coarse, 20% eye fine offset)
        target_x = np.clip(head_x + (iris_rel_x * .5), 0.0, 1.0) * SCREEN_W
        target_y = np.clip(head_y + (iris_rel_y * .5), 0.0, 1.0) * SCREEN_H

        speed = np.clip(abs(np.hypot(target_x-smooth_x, target_y-smooth_y)),0,200)

        ALPHA = np.interp(speed,[0,200],[0.02,0.1])



        # 4. SMOOTHING (Exponential Moving Average)
        smooth_x = (ALPHA * target_x) + ((1 - ALPHA) * smooth_x)
        smooth_y = (ALPHA * target_y) + ((1 - ALPHA) * smooth_y)

        # if abs(np.hypot(smooth_x-anchor_x,smooth_y-anchor_y)) < 8:
        #     smooth_x,smooth_y = anchor_x,anchor_y
        # else:
        #     anchor_x,anchor_y = smooth_x,smooth_y 


        if tracking_active and ear > 0.3:
            pg.moveTo(int(smooth_x), int(smooth_y))

        # Draw diagnostic circle on nose & iris
        cv2.circle(frame, (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)), 4, (0, 255, 0), -1)
        cv2.circle(frame, (int(lm[LEFT_IRIS].x * w), int(lm[LEFT_IRIS].y * h)), 3, (0, 255, 255), -1)

    status = "ACTIVE (Press SPACE to pause)" if tracking_active else "PAUSED (Press SPACE to start)"
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Pilot MVP", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        tracking_active = not tracking_active

cap.release()
cv2.destroyAllWindows()
