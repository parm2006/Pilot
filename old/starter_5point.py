"""
Pilot — Clean Reference Baseline (Decoupled Eye Range + 5-Point Bilinear Quadrant Map)
Run with: `uv run python starter.py`

Workflow:
  1. Hold 'E' (~2 sec) while rolling your eyes Left / Right / Up / Down (head still)
     -> Measures your personal eye socket bounds at full frame rate (GetAsyncKeyState).
  2. Look at the 5 Yellow Bullseyes (CENTER -> TL -> TR -> BR -> BL) and press 'C'
     -> Locks the 4-Quadrant Bilinear Map around CENTER so every corner hits (0.02, 0.98)
        and neither head nor eyes can ever invert direction!
  3. Press 'SPACE' to toggle tracking | 'R' to reset | 'Q' to quit
"""

from collections import deque
import ctypes
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg

pg.FAILSAFE = True
pg.PAUSE = 0
SCREEN_W, SCREEN_H = pg.size()


def is_key_held(letter: str) -> bool:
    """Reads physical key state on Windows with zero auto-repeat delay."""
    return (ctypes.windll.user32.GetAsyncKeyState(ord(letter.upper())) & 0x8000) != 0


# MediaPipe Landmarks
NOSE_TIP = 1
LEFT_IRIS, RIGHT_IRIS = 468, 473
L_IRIS_BOTTOM, R_IRIS_BOTTOM = 472, 477  # Bottom iris rim (immune to upper eyelid movement)
L_EYE_INNER, L_EYE_OUTER = 133, 33
R_EYE_INNER, R_EYE_OUTER = 362, 263
L_EYE_TOP, L_EYE_BOTTOM = 159, 145

# Anatomically scaled 3D Skull Model (in mm) for a flipped frame (+X Right, +Y Down, +Z Deeper)
FACE_3D_IDS = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        # Nose tip (1)
    [0.0, 68.0, 18.0],      # Chin (152)
    [-46.0, -34.0, 42.0],   # Left eye outer corner (33)
    [46.0, -34.0, 42.0],    # Right eye outer corner (263)
    [-28.0, 30.0, 38.0],    # Left mouth corner (61)
    [28.0, 30.0, 38.0],     # Right mouth corner (291)
], dtype=np.float64)

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

# --- TWO-STAGE GAZE BUBBLE + CURSOR FOLLOWER SMOOTHING ---
smooth_x, smooth_y = SCREEN_W / 2.0, SCREEN_H / 2.0
cursor_x, cursor_y = smooth_x, smooth_y
bubble_radius = 38.0
MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS = 12.0, 60.0
RADIUS_ALPHA = 0.20
CURSOR_ALPHA = 0.45

# Rolling buffers for sensor de-noising
eye_buf_x, eye_buf_y = deque(maxlen=6), deque(maxlen=6)
head_buf_x, head_buf_y = deque(maxlen=6), deque(maxlen=6)
ear_baseline_buf = deque(maxlen=50)
last_good_eye_x, last_good_eye_y = 0.0, -0.07

# --- STEP 1: EYE RANGE RECORDER (Hold 'E') ---
eye_samples_x, eye_samples_y = [], []
# Neutral eye_x ~ 0.0, neutral eye_y ~ -0.07 (relative to eye corner axis)
eye_bounds = [-0.08, 0.08, -0.15, 0.01]  # [min_x, max_x, min_y, max_y]

# --- STEP 2: 5-POINT BILINEAR QUADRANT CALIBRATION (Press 'C' 5 times) ---
CALIB_TARGETS = [
    ("CENTER", (0.50, 0.50)),
    ("TOP-LEFT", (0.02, 0.02)),
    ("TOP-RIGHT", (0.98, 0.02)),
    ("BOTTOM-RIGHT", (0.98, 0.98)),
    ("BOTTOM-LEFT", (0.02, 0.98)),
]

calib_pts = []
last_calib_time = 0.0
calibrated = False

# Bilinear 4-Quadrant Anchors:
# Center (cx, cy) defaults to straight-ahead neutral (-0.14 Y from head pose, 0.0 from eyes)
# so straight ahead defaults to screen center (0.50, 0.50)
quad_map = {
    "cx": 0.0, "cy": -0.14,
    "tl_x": -0.35, "tr_x": 0.35, "br_x": 0.35, "bl_x": -0.35,
    "tl_y": -0.40, "tr_y": -0.40, "br_y": 0.15, "bl_y": 0.15,
}

WINDOW_NAME = "Pilot 3D - Reference Handoff Build"


def set_fullscreen(is_full: bool):
    if is_full:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, int(SCREEN_W * 0.55), int(SCREEN_H * 0.55))


def extract_corner_aligned_iris(lm, inner_id, outer_id, iris_id, is_left_eye: bool):
    """
    Projects the true iris center onto the anatomical eye-corner axis (u, v)
    so head roll and perspective tilt NEVER invert vertical eye_y.
    """
    # Vector pointing from Left-most corner on image to Right-most corner on image
    if is_left_eye:
        p_left, p_right = lm[outer_id], lm[inner_id]   # Outer(33) is left of Inner(133)
    else:
        p_left, p_right = lm[inner_id], lm[outer_id]   # Inner(362) is left of Outer(263)

    dx = p_right.x - p_left.x
    dy = p_right.y - p_left.y
    width = np.hypot(dx, dy) + 1e-6

    # Unit basis: u = horizontal along corners (+X right), v = perpendicular (+Y down)
    ux, uy = dx / width, dy / width
    vx, vy = -uy, ux

    mid_x = (p_left.x + p_right.x) / 2.0
    mid_y = (p_left.y + p_right.y) / 2.0

    # True iris center
    ix = lm[iris_id].x
    iy = lm[iris_id].y

    rel_x = (ix - mid_x) / width
    rel_y = (iy - mid_y) / width

    # Dot product onto corner basis (ux, uy) and (vx, vy)
    eye_x = rel_x * ux + rel_y * uy
    eye_y = rel_x * vx + rel_y * vy
    return eye_x, eye_y, width


try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    set_fullscreen(True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        valid_sample = False
        combined_x, combined_y = 0.0, 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # 1. BLINK & EYEBROW-RAISE GUARD (EAR)
            l_w = np.hypot(lm[L_EYE_INNER].x - lm[L_EYE_OUTER].x, lm[L_EYE_INNER].y - lm[L_EYE_OUTER].y) + 1e-6
            l_h = np.hypot(lm[L_EYE_TOP].x - lm[L_EYE_BOTTOM].x, lm[L_EYE_TOP].y - lm[L_EYE_BOTTOM].y) + 1e-6
            ear = l_h / l_w

            if 0.24 < ear < 0.42:
                ear_baseline_buf.append(ear)
            neutral_ear = float(np.median(ear_baseline_buf)) if ear_baseline_buf else 0.32
            valid_sample = (ear > 0.25) and (ear < neutral_ear * 1.24)

            # 2. CORNER-PROJECTED BINOCULAR EYE VECTOR (Roll-immune & Eyelid-immune)
            if valid_sample:
                lx, ly, _ = extract_corner_aligned_iris(lm, L_EYE_INNER, L_EYE_OUTER, LEFT_IRIS, True)
                rx, ry, _ = extract_corner_aligned_iris(lm, R_EYE_INNER, R_EYE_OUTER, RIGHT_IRIS, False)
                eye_buf_x.append((lx + rx) / 2.0)
                eye_buf_y.append((ly + ry) / 2.0)
                last_good_eye_x = float(np.median(eye_buf_x))
                last_good_eye_y = float(np.median(eye_buf_y))

            eye_x, eye_y = last_good_eye_x, last_good_eye_y

            # 3. 3D HEAD POSE VECTOR (Rotation + Perspective Position)
            face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_3D_IDS], dtype=np.float64)
            focal_len = w * 0.72
            cam_matrix = np.array([[focal_len, 0, w / 2.0], [0, focal_len, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success_pnp, rvec, tvec = cv2.solvePnP(
                FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
            )

            if success_pnp:
                rmat, _ = cv2.Rodrigues(rvec)
                raw_x, raw_y, raw_z = tvec[0, 0], tvec[1, 0], max(float(tvec[2, 0]), 200.0)
                head_ray_x = -rmat[0, 2] + (raw_x / raw_z) * 0.40
                head_ray_y = -rmat[1, 2] + (raw_y / raw_z) * 0.40
                head_buf_x.append(head_ray_x)
                head_buf_y.append(head_ray_y)

            head_x = float(np.mean(head_buf_x)) if head_buf_x else 0.0
            head_y = float(np.mean(head_buf_y)) if head_buf_y else 0.0

            # Record Eye Range while 'E' is physically held down
            if is_key_held("e") and valid_sample:
                eye_samples_x.append(eye_x)
                eye_samples_y.append(eye_y)
                if len(eye_samples_x) >= 8:
                    eye_bounds = [
                        float(np.percentile(eye_samples_x, 5)),
                        float(np.percentile(eye_samples_x, 95)),
                        float(np.percentile(eye_samples_y, 5)),
                        float(np.percentile(eye_samples_y, 95)),
                    ]

            # Normalize eye vector with range clamping so outliers never explode
            e_min_x, e_max_x, e_min_y, e_max_y = eye_bounds
            eye_norm_x = float(np.clip(((eye_x - e_min_x) / (e_max_x - e_min_x + 1e-6) - 0.5) * 0.35, -0.25, 0.25))
            eye_norm_y = float(np.clip(((eye_y - e_min_y) / (e_max_y - e_min_y + 1e-6) - 0.5) * 0.35, -0.25, 0.25))

            # Strictly positive fusion: Head + Eye ALWAYS move in the same direction (never fight!)
            combined_x = head_x + eye_norm_x
            combined_y = head_y + eye_norm_y

            # 4. 4-QUADRANT BILINEAR INTERPOLATION AROUND CALIBRATED CENTER
            cx, cy = quad_map["cx"], quad_map["cy"]
            dx = combined_x - cx
            dy = combined_y - cy

            # Horizontal reach (Left vs Right of Center)
            left_span = max(cx - min(quad_map["tl_x"], quad_map["bl_x"]), 0.04)
            right_span = max(max(quad_map["tr_x"], quad_map["br_x"]) - cx, 0.04)
            norm_x = 0.50 + (0.48 * (dx / left_span) if dx < 0 else 0.48 * (dx / right_span))

            # Vertical reach interpolated horizontally between Left-side Y limits and Right-side Y limits!
            # (Fixes the bug where looking down on the left side had a different vertical offset than the right side)
            hx_ratio = float(np.clip(norm_x, 0.0, 1.0))
            top_limit_at_x = (1.0 - hx_ratio) * quad_map["tl_y"] + hx_ratio * quad_map["tr_y"]
            bot_limit_at_x = (1.0 - hx_ratio) * quad_map["bl_y"] + hx_ratio * quad_map["br_y"]

            top_span = max(cy - top_limit_at_x, 0.03)
            bot_span = max(bot_limit_at_x - cy, 0.03)
            norm_y = 0.50 + (0.48 * (dy / top_span) if dy < 0 else 0.48 * (dy / bot_span))

            norm_x = float(np.clip(norm_x, 0.0, 1.0))
            norm_y = float(np.clip(norm_y, 0.0, 1.0))

            target_x = norm_x * (SCREEN_W - 1)
            target_y = norm_y * (SCREEN_H - 1)

            # 5. TWO-STAGE SMOOTHING + BUBBLE CLAMP
            speed = np.hypot(target_x - smooth_x, target_y - smooth_y)
            bubble_alpha = np.interp(speed, [0, 220], [0.03, 0.28])
            smooth_x = bubble_alpha * target_x + (1.0 - bubble_alpha) * smooth_x
            smooth_y = bubble_alpha * target_y + (1.0 - bubble_alpha) * smooth_y

            target_radius = np.interp(speed, [0, 180], [MIN_BUBBLE_RADIUS, MAX_BUBBLE_RADIUS])
            bubble_radius = RADIUS_ALPHA * target_radius + (1.0 - RADIUS_ALPHA) * bubble_radius

            dist_to_bubble = np.hypot(smooth_x - cursor_x, smooth_y - cursor_y)
            active_cursor_alpha = CURSOR_ALPHA if dist_to_bubble > bubble_radius else CURSOR_ALPHA * 0.35
            cursor_x = active_cursor_alpha * smooth_x + (1.0 - active_cursor_alpha) * cursor_x
            cursor_y = active_cursor_alpha * smooth_y + (1.0 - active_cursor_alpha) * cursor_y

            new_dist = np.hypot(cursor_x - smooth_x, cursor_y - smooth_y)
            if new_dist > bubble_radius and new_dist > 1e-6:
                scale = bubble_radius / new_dist
                cursor_x = smooth_x + (cursor_x - smooth_x) * scale
                cursor_y = smooth_y + (cursor_y - smooth_y) * scale

            # Cursor moving code commented out
            # if tracking_active and valid_sample:
            #     pg.moveTo(int(cursor_x), int(cursor_y))

            # Draw the 5 calibration targets
            active_idx = len(calib_pts) if (not calibrated or calib_pts) else -1
            for idx, (name, (tx, ty)) in enumerate(CALIB_TARGETS):
                px, py = int(tx * (w - 1)), int(ty * (h - 1))
                if idx == active_idx:
                    cv2.circle(frame, (px, py), 18, (0, 255, 255), 3)
                    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (px, py), 6, (255, 80, 140), -1)

            # Draw moving gaze bubble (dynamic radius) and cursor circle
            bx, by = int((smooth_x / SCREEN_W) * w), int((smooth_y / SCREEN_H) * h)
            cx, cy = int((cursor_x / SCREEN_W) * w), int((cursor_y / SCREEN_H) * h)
            hud_bubble_radius = max(6, int((bubble_radius / SCREEN_W) * w))

            # Outer gaze bubble
            cv2.circle(frame, (bx, by), hud_bubble_radius, (255, 255, 0), 2)
            # Inner circle / cursor
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)

            cv2.putText(
                frame,
                f"Head:({head_x:+.3f},{head_y:+.3f}) | EyeNorm:({eye_norm_x:+.3f},{eye_norm_y:+.3f}) | E_pts:{len(eye_samples_x)}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2,
            )

        status = "ACTIVE (SPACE: pause)" if tracking_active else "PAUSED (SPACE: start | R: reset)"
        cv2.putText(frame, status, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

        if len(eye_samples_x) < 15:
            prompt = "STEP 1: Hold 'E' & roll eyes Left / Right / Up / Down (keep head still)"
        elif not calibrated or calib_pts:
            step_name = CALIB_TARGETS[len(calib_pts)][0]
            prompt = f"STEP 2: Look at {step_name} (Yellow Bullseye) & press 'C' ({len(calib_pts)}/5)"
        else:
            prompt = "CALIBRATED! Bubble/Circle = Gaze | SPACE = Toggle Tracking | R = Reset"
        cv2.putText(frame, prompt, (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if calibrated and not calib_pts:
                tracking_active = not tracking_active
                if tracking_active:
                    set_fullscreen(False)
        elif key == ord("r"):
            eye_samples_x.clear()
            eye_samples_y.clear()
            calib_pts.clear()
            calibrated = False
            tracking_active = False
            quad_map = {
                "cx": 0.0, "cy": -0.14,
                "tl_x": -0.35, "tr_x": 0.35, "br_x": 0.35, "bl_x": -0.35,
                "tl_y": -0.40, "tr_y": -0.40, "br_y": 0.15, "bl_y": 0.15,
            }
            eye_bounds = [-0.08, 0.08, -0.15, 0.01]
            set_fullscreen(True)
            print("Reset all calibrations.")
        elif key == ord("c"):
            now = time.perf_counter()
            if valid_sample and now - last_calib_time > 0.45:
                last_calib_time = now
                calib_pts.append((combined_x, combined_y))
                step_name = CALIB_TARGETS[len(calib_pts) - 1][0]
                print(f"Captured {step_name} ({len(calib_pts)}/5): ({combined_x:+.3f}, {combined_y:+.3f})")

                if len(calib_pts) == 5:
                    quad_map = {
                        "cx": calib_pts[0][0],  "cy": calib_pts[0][1],
                        "tl_x": calib_pts[1][0], "tl_y": calib_pts[1][1],
                        "tr_x": calib_pts[2][0], "tr_y": calib_pts[2][1],
                        "br_x": calib_pts[3][0], "br_y": calib_pts[3][1],
                        "bl_x": calib_pts[4][0], "bl_y": calib_pts[4][1],
                    }
                    calibrated = True
                    smooth_x, smooth_y = SCREEN_W / 2.0, SCREEN_H / 2.0
                    cursor_x, cursor_y = smooth_x, smooth_y
                    print("5-Point Bilinear Quadrant Map Locked!", quad_map)
                    calib_pts.clear()
finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()