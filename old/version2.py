import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg

pg.FAILSAFE = True
pg.PAUSE = 0
SCREEN_W, SCREEN_H = pg.size()

NOSE_TIP = 1
LEFT_IRIS = 468
L_EYE_INNER, L_EYE_OUTER = 133, 33
L_EYE_TOP, L_EYE_BOTTOM = 159, 145

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
tracking_active = False

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
cursor_x, cursor_y = smooth_x, smooth_y

bubble_radius = 40.0
MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS = 14.0, 65.0
RADIUS_ALPHA = 0.20
CURSOR_ALPHA = 0.45  # Snappier outside the bubble so the cursor stays inside

mouse_down = False
last_good_iris_y = 0.0
last_good_iris_x = 0.0
blink_counter = 0

while cap.isOpened():
    success, frame = cap.read()
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

        ear = eye_height / eye_width

        if ear < 0.3:  # Your natural open EAR is ~0.40
            iris_rel_x = last_good_iris_x
            iris_rel_y = last_good_iris_y
            blink_counter += 1

            if tracking_active and blink_counter == 7:
                pg.mouseDown(button="left")
                mouse_down = True
        else:
            iris_rel_x = (lm[LEFT_IRIS].x - lm[L_EYE_OUTER].x) / eye_width - 0.5
            iris_rel_y = (lm[LEFT_IRIS].y - lm[L_EYE_TOP].y) / eye_height - 0.5
            last_good_iris_x = iris_rel_x
            last_good_iris_y = iris_rel_y
            blink_counter = 0

            if mouse_down:
                mouse_down = False
                pg.mouseUp(button="left")

        cv2.putText(
            frame,
            f"EAR: {round(ear, 2)} | Blink: {blink_counter} | R: {int(bubble_radius)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # 3. FUSE HEAD + EYE
        target_x = np.clip(head_x + (iris_rel_x * 0.5), 0.0, 1.0) * SCREEN_W
        target_y = np.clip(head_y + (iris_rel_y * 0.5), 0.0, 1.0) * SCREEN_H

        # 4. BUBBLE SMOOTHING + DYNAMIC RADIUS
        speed = np.hypot(target_x - smooth_x, target_y - smooth_y)
        bubble_alpha = np.interp(speed, [0, 200], [0.03, 0.25])

        smooth_x = (bubble_alpha * target_x) + ((1 - bubble_alpha) * smooth_x)
        smooth_y = (bubble_alpha * target_y) + ((1 - bubble_alpha) * smooth_y)

        target_radius = np.interp(speed, [0, 180], [MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS])
        bubble_radius = (RADIUS_ALPHA * target_radius) + ((1 - RADIUS_ALPHA) * bubble_radius)

        # 5. CURSOR ABSTRACTION LAYER (keeps cursor inside the bubble while moving smoothly)
        dist_to_bubble = np.hypot(smooth_x - cursor_x, smooth_y - cursor_y)
        active_cursor_alpha = CURSOR_ALPHA if dist_to_bubble > bubble_radius else CURSOR_ALPHA * 0.35

        cursor_x = (active_cursor_alpha * smooth_x) + ((1 - active_cursor_alpha) * cursor_x)
        cursor_y = (active_cursor_alpha * smooth_y) + ((1 - active_cursor_alpha) * cursor_y)

        # Clamp cursor inside bubble_radius if it's still outside after the step
        new_dist = np.hypot(cursor_x - smooth_x, cursor_y - smooth_y)
        if new_dist > bubble_radius and new_dist > 1e-6:
            scale = bubble_radius / new_dist
            cursor_x = smooth_x + (cursor_x - smooth_x) * scale
            cursor_y = smooth_y + (cursor_y - smooth_y) * scale

        if tracking_active and (ear >= 0.3 or blink_counter >= 15):
            pg.moveTo(int(cursor_x), int(cursor_y))

        # Draw diagnostic circles on nose, iris, bubble, and cursor
        cv2.circle(frame, (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)), 4, (0, 255, 0), -1)
        cv2.circle(frame, (int(lm[LEFT_IRIS].x * w), int(lm[LEFT_IRIS].y * h)), 3, (0, 255, 255), -1)

        bx, by = int((smooth_x / SCREEN_W) * w), int((smooth_y / SCREEN_H) * h)
        cx, cy = int((cursor_x / SCREEN_W) * w), int((cursor_y / SCREEN_H) * h)
        hud_radius = max(4, int((bubble_radius / SCREEN_W) * w))

        cv2.circle(frame, (bx, by), hud_radius, (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

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