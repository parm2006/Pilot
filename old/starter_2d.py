"""Run with `uv run python starter_2d.py`. SPACE: pause, C: calibrate, Q: quit."""

import time

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

CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
calib_pts = []
calib_bounds = [0.0, 1.0, 0.0, 1.0]
last_calib_time = 0.0
calibrated = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
tracking_active = False

smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
cursor_x, cursor_y = smooth_x, smooth_y
bubble_radius = 40.0
MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS = 14.0, 65.0
RADIUS_ALPHA = 0.20
CURSOR_ALPHA = 0.45
last_good_iris_x, last_good_iris_y = 0.0, 0.0

WINDOW_NAME = "Pilot 2D"

try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(SCREEN_W * 0.7), int(SCREEN_H * 0.7))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        valid_sample = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Original 2D nose-position mapping from old/version2.py.
            head_x = np.clip((lm[NOSE_TIP].x - 0.35) / 0.30, 0.0, 1.0)
            head_y = np.clip((lm[NOSE_TIP].y - 0.35) / 0.30, 0.0, 1.0)

            eye_width = abs(lm[L_EYE_INNER].x - lm[L_EYE_OUTER].x) + 1e-6
            eye_height = abs(lm[L_EYE_BOTTOM].y - lm[L_EYE_TOP].y) + 1e-6
            ear = eye_height / eye_width
            valid_sample = ear >= 0.3

            if valid_sample:
                iris_rel_x = (lm[LEFT_IRIS].x - lm[L_EYE_OUTER].x) / eye_width - 0.5
                iris_rel_y = (lm[LEFT_IRIS].y - lm[L_EYE_TOP].y) / eye_height - 0.5
                last_good_iris_x = iris_rel_x
                last_good_iris_y = iris_rel_y
            else:
                iris_rel_x = last_good_iris_x
                iris_rel_y = last_good_iris_y

            # Capture the combined reading before the final screen clamp.
            raw_gaze_x = head_x + iris_rel_x * 0.5
            raw_gaze_y = head_y + iris_rel_y * 0.5

            min_x, max_x, min_y, max_y = calib_bounds
            norm_x = np.clip((raw_gaze_x - min_x) / (max_x - min_x), 0.0, 1.0)
            norm_y = np.clip((raw_gaze_y - min_y) / (max_y - min_y), 0.0, 1.0)
            target_x = norm_x * (SCREEN_W - 1)
            target_y = norm_y * (SCREEN_H - 1)

            # Original two-stage smoothing, with the bubble kept invisible.
            speed = np.hypot(target_x - smooth_x, target_y - smooth_y)
            bubble_alpha = np.interp(speed, [0, 200], [0.03, 0.25])
            smooth_x = bubble_alpha * target_x + (1 - bubble_alpha) * smooth_x
            smooth_y = bubble_alpha * target_y + (1 - bubble_alpha) * smooth_y

            target_radius = np.interp(speed, [0, 180], [MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS])
            bubble_radius = RADIUS_ALPHA * target_radius + (1 - RADIUS_ALPHA) * bubble_radius

            dist_to_bubble = np.hypot(smooth_x - cursor_x, smooth_y - cursor_y)
            active_cursor_alpha = CURSOR_ALPHA if dist_to_bubble > bubble_radius else CURSOR_ALPHA * 0.35
            cursor_x = active_cursor_alpha * smooth_x + (1 - active_cursor_alpha) * cursor_x
            cursor_y = active_cursor_alpha * smooth_y + (1 - active_cursor_alpha) * cursor_y

            new_dist = np.hypot(cursor_x - smooth_x, cursor_y - smooth_y)
            if new_dist > bubble_radius and new_dist > 1e-6:
                scale = bubble_radius / new_dist
                cursor_x = smooth_x + (cursor_x - smooth_x) * scale
                cursor_y = smooth_y + (cursor_y - smooth_y) * scale

            if tracking_active and valid_sample:
                pg.moveTo(int(cursor_x), int(cursor_y))

            cv2.putText(
                frame,
                f"EAR: {ear:.2f} | Raw gaze: {raw_gaze_x:.3f}, {raw_gaze_y:.3f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )

        status = "ACTIVE (SPACE to pause)" if tracking_active else "PAUSED (SPACE to start)"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not calibrated or calib_pts:
            prompt = f"Look {CORNER_NAMES[len(calib_pts)]} & press C ({len(calib_pts)}/4)"
        else:
            prompt = "Calibrated | C to recalibrate | Q to quit"
        cv2.putText(frame, prompt, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not calib_pts:
                tracking_active = not tracking_active
        elif key == ord("c"):
            now = time.perf_counter()
            if valid_sample and now - last_calib_time > 0.6:
                tracking_active = False
                last_calib_time = now
                calib_pts.append((raw_gaze_x, raw_gaze_y))
                print(f"Captured {CORNER_NAMES[len(calib_pts) - 1]}: {raw_gaze_x:.3f}, {raw_gaze_y:.3f}")

                if len(calib_pts) == 4:
                    xs = [p[0] for p in calib_pts]
                    ys = [p[1] for p in calib_pts]
                    bounds = [min(xs), max(xs), min(ys), max(ys)]
                    if bounds[1] - bounds[0] > 1e-6 and bounds[3] - bounds[2] > 1e-6:
                        calib_bounds = bounds
                        calibrated = True
                        smooth_x, smooth_y = SCREEN_W / 2, SCREEN_H / 2
                        cursor_x, cursor_y = smooth_x, smooth_y
                        bubble_radius = 40.0
                        print("New calib_bounds:", calib_bounds, "| SPACE to start")
                    else:
                        print("Calibration readings need more spread. Try the four corners again.")
                    calib_pts = []
finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
