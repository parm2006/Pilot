import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg

pg.FAILSAFE = True
pg.PAUSE = 0
SCREEN_W, SCREEN_H = pg.size()

# Landmarks: Nose, Left Eye, Right Eye, and Left Iris Contour (468 is center, 469/471 are horizontal edges)
NOSE_TIP = 1
LEFT_IRIS, RIGHT_IRIS = 468, 473
L_IRIS_LEFT, L_IRIS_RIGHT = 469, 471
L_EYE_INNER, L_EYE_OUTER = 133, 33
R_EYE_INNER, R_EYE_OUTER = 362, 263
L_EYE_TOP, L_EYE_BOTTOM = 159, 145

# 3D Skull Model (in mm) for a flipped frame (+X Right, +Y Down, +Z Forward away from camera)
# Order: [Nose(1), Chin(152), L_Eye_Outer(33), R_Eye_Outer(263), L_Mouth(61), R_Mouth(291)]
FACE_3D_IDS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        # Nose tip (1)
    [0.0, 68.0, 18.0],      # Chin (152) - below nose (+Y), deeper (+Z)
    [-46.0, -34.0, 42.0],   # Landmark 33 (Left side of flipped image = -X)
    [46.0, -34.0, 42.0],    # Landmark 263 (Right side of flipped image = +X)
    [-28.0, 30.0, 38.0],    # Landmark 61 (Left mouth on flipped image = -X)
    [28.0, 30.0, 38.0],     # Landmark 291 (Right mouth on flipped image = +X)
], dtype=np.float64)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tracking_active = False

# Calm Two-Stage Smoothing (Bubble + Cursor) from your 2D version
bubble_x, bubble_y = SCREEN_W / 2.0, SCREEN_H / 2.0
cursor_x, cursor_y = bubble_x, bubble_y

bubble_radius = 40.0
MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS = 14.0, 65.0
RADIUS_ALPHA = 0.20
CURSOR_ALPHA = 0.14  # Calm cursor glide

# Smoothed 3D head position (in mm) so solvePnP depth noise never shakes the ray
smooth_tvec = np.array([0.0, 0.0, 600.0], dtype=np.float64)
center_hit_x, center_hit_y = 0.0, 0.0

# Virtual 3D Screen Plane bounds in mm (increase to slow down sensitivity, decrease to speed up)
VIRTUAL_SCREEN_W_MM = 550.0
VIRTUAL_SCREEN_H_MM = 320.0
EYE_SPHERE_GAIN = 1.8  # How strongly eyeball rotation steers the 3D ray relative to skull

WINDOW_NAME = "Pilot MVP"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, int(SCREEN_W * 0.7), int(SCREEN_H * 0.7))

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

        # --- STEP 1: 3D EYEBALL SPHERE VECTOR (LEFT EYE) + 3D HEAD POSE ---
        # Measure socket width & rigid corner center
        l_width = abs(lm[L_EYE_INNER].x - lm[L_EYE_OUTER].x) + 1e-6
        l_mid_x = (lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2.0
        l_mid_y = (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2.0

        # Iris displacement on the 3D eyeball sphere (normalized by eyeball radius ~ 0.42 * eye_width)
        eyeball_radius_norm = l_width * 0.42
        eye_rot_x = ((lm[LEFT_IRIS].x - l_mid_x) / eyeball_radius_norm) * EYE_SPHERE_GAIN
        eye_rot_y = ((lm[LEFT_IRIS].y - l_mid_y) / eyeball_radius_norm) * EYE_SPHERE_GAIN

        # Solve 3D Skull Pose
        face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_3D_IDS], dtype=np.float64)
        focal_len = w * 0.72
        cam_matrix = np.array([[focal_len, 0, w / 2.0], [0, focal_len, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success_pnp, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        hit_x_mm, hit_y_mm = center_hit_x, center_hit_y
        raw_x = raw_y = raw_z = 0.0

        if success_pnp:
            # 1. Read raw signed translation (no abs, no clip, no smoothing)
            raw_x = tvec[0, 0]
            raw_y = tvec[1, 0]
            raw_z = tvec[2, 0]

            # 2. Reproject the 6 3D model points back onto the 2D camera frame
            proj_pts, _ = cv2.projectPoints(FACE_3D_MODEL, rvec, tvec, cam_matrix, dist_coeffs)
            proj_pts = proj_pts.reshape(-1, 2)

            # 3. Draw observed MediaPipe landmarks (Green) vs. Reprojected 3D model (Red)
            for (obs_x, obs_y), (prj_x, prj_y) in zip(face_2d, proj_pts):
                cv2.circle(frame, (int(obs_x), int(obs_y)), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(prj_x), int(prj_y)), 3, (0, 0, 255), -1)

        rel_hit_x, rel_hit_y = 0.0, 0.0

        cv2.putText(
            frame,
            f"Raw tvec: X={raw_x:.1f} | Y={raw_y:.1f} | Z={raw_z:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # --- STEP 2: MAP 3D PLANE INTERSECTION TO GAZE BUBBLE ---
        norm_x = np.clip(0.5 + (rel_hit_x / VIRTUAL_SCREEN_W_MM), 0.0, 1.0)
        norm_y = np.clip(0.5 + (rel_hit_y / VIRTUAL_SCREEN_H_MM), 0.0, 1.0)

        target_x = norm_x * SCREEN_W
        target_y = norm_y * SCREEN_H

        # Calm velocity-adaptive smoothing [0.02, 0.10]
        speed = np.hypot(target_x - bubble_x, target_y - bubble_y)
        bubble_alpha = np.interp(speed, [0, 200], [0.02, 0.10])

        bubble_x = (bubble_alpha * target_x) + ((1.0 - bubble_alpha) * bubble_x)
        bubble_y = (bubble_alpha * target_y) + ((1.0 - bubble_alpha) * bubble_y)

        target_radius = np.interp(speed, [0, 180], [MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS])
        bubble_radius = (RADIUS_ALPHA * target_radius) + ((1.0 - RADIUS_ALPHA) * bubble_radius)

        # --- STEP 3: GLIDE CURSOR TO BUBBLE CENTER ---
        dist_to_bubble = np.hypot(bubble_x - cursor_x, bubble_y - cursor_y)
        active_cursor_alpha = CURSOR_ALPHA if dist_to_bubble > bubble_radius else CURSOR_ALPHA * 0.2

        cursor_x = (active_cursor_alpha * bubble_x) + ((1.0 - active_cursor_alpha) * cursor_x)
        cursor_y = (active_cursor_alpha * bubble_y) + ((1.0 - active_cursor_alpha) * cursor_y)

        if tracking_active:
            pg.moveTo(int(cursor_x), int(cursor_y))

    status = "ACTIVE (SPACE to pause)" if tracking_active else "PAUSED (Look center & press SPACE)"
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        tracking_active = not tracking_active
        if tracking_active:
            # Lock current 3D gaze-plane intersection as screen center (0.5, 0.5)
            center_hit_x = hit_x_mm
            center_hit_y = hit_y_mm

cap.release()
cv2.destroyAllWindows()