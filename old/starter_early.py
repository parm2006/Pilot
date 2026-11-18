import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
import time
CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
last_calib_time = 0.0

pg.FAILSAFE = False
# pg.FAILSAFE = True
pg.PAUSE = 0
SCREEN_W, SCREEN_H = pg.size()

# Landmarks: Nose, Left Eye, Right Eye, and Iris Contours
NOSE_TIP = 1
LEFT_IRIS, RIGHT_IRIS = 468, 473
L_IRIS_LEFT, L_IRIS_RIGHT = 469, 471
L_EYE_INNER, L_EYE_OUTER = 133, 33
R_EYE_INNER, R_EYE_OUTER = 362, 263
L_EYE_TOP, L_EYE_BOTTOM = 159, 145
R_EYE_TOP, R_EYE_BOTTOM = 386, 374

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

# Expand logical screen (1.20 = 120% logical size)
# Pushes the difficult >95% edge reach outside physical bounds
LOGICAL_SCREEN_SCALE = 1.20
LOGICAL_W = SCREEN_W * LOGICAL_SCREEN_SCALE
LOGICAL_H = SCREEN_H * LOGICAL_SCREEN_SCALE
LOGICAL_OFFSET_X = (LOGICAL_W - SCREEN_W) / 2.0
LOGICAL_OFFSET_Y = (LOGICAL_H - SCREEN_H) / 2.0

# Calibration step targets:
# Step 0: Center lock
# Steps 1-4: Comfortable reach (TL, TR, BR, BL) - do NOT strain
# Steps 5-8: Full reach (TL, TR, BR, BL) - move head until bubble reaches corner
CALIB_STEPS = [
    ("CENTER", "STEP 1/9: Look CENTER & press 'C' to lock straight ahead", (0.50, 0.50)),
    ("COMFORT_TL", "STEP 2/9 [COMFORT]: Look comfortably TOP-LEFT & press 'C'", (0.18, 0.18)),
    ("COMFORT_TR", "STEP 3/9 [COMFORT]: Look comfortably TOP-RIGHT & press 'C'", (0.82, 0.18)),
    ("COMFORT_BR", "STEP 4/9 [COMFORT]: Look comfortably BOTTOM-RIGHT & press 'C'", (0.82, 0.82)),
    ("COMFORT_BL", "STEP 5/9 [COMFORT]: Look comfortably BOTTOM-LEFT & press 'C'", (0.18, 0.82)),
    ("FULL_TL", "STEP 6/9 [FULL REACH]: Steer bubble to TOP-LEFT corner & press 'C'", (0.02, 0.02)),
    ("FULL_TR", "STEP 7/9 [FULL REACH]: Steer bubble to TOP-RIGHT corner & press 'C'", (0.98, 0.02)),
    ("FULL_BR", "STEP 8/9 [FULL REACH]: Steer bubble to BOTTOM-RIGHT corner & press 'C'", (0.98, 0.98)),
    ("FULL_BL", "STEP 9/9 [FULL REACH]: Steer bubble to BOTTOM-LEFT corner & press 'C'", (0.02, 0.98)),
]

calib_step = 0
is_calibrated = False
calib_center = np.array([0.0, -0.14])
comfort_vectors = [np.array([-0.20, -0.15]), np.array([0.20, -0.15]), np.array([0.20, 0.15]), np.array([-0.20, 0.15])]
full_vectors = [np.array([-0.35, -0.24]), np.array([0.35, -0.24]), np.array([0.35, 0.24]), np.array([-0.35, 0.24])]
# Difference scale factors [TL, TR, BR, BL] (default to 1.75x)
scale_factors = [(1.75, 1.75), (1.75, 1.75), (1.75, 1.75), (1.75, 1.75)]
full_spans = [0.35, 0.35, 0.24, 0.24]  # [left, right, top, bottom]

# Eye weights: X (horizontal) and Y (vertical downward/upward boost)
EYE_WEIGHT_X = 2.5
EYE_WEIGHT_Y = 3.5

# Neutral vertical iris offset (pupil center naturally sits ~0.065 above canthus midpoint)
IRIS_NEUTRAL_Y_OFFSET = -0.065

# Blink detection threshold (0.20 avoids locking downward gaze as a blink)
BLINK_THRESHOLD = 0.20

WINDOW_NAME = "Pilot MVP"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, int(SCREEN_W * 0.7), int(SCREEN_H * 0.7))

# Console / diagnostic log file
LOG_FILE_PATH = "pilot_diagnostic.log"
log_file = open(LOG_FILE_PATH, "a", buffering=1, encoding="utf-8")
log_file.write(f"\n{'='*70}\n[START] Pilot Diagnostic Session: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"Screen: {SCREEN_W}x{SCREEN_H} | Logical Scale: {LOGICAL_SCREEN_SCALE:.2f}\n"
               f"Center: {calib_center} | Spans: {full_spans}\n{'='*70}\n")
print(f"Logging diagnostic telemetry to: {LOG_FILE_PATH}")

last_good_iris_x, last_good_iris_y = 0.0, 0.0
raw_gaze_x, raw_gaze_y = 0.0, -0.14
frame_count = 0

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

        # --- STEP 1: 3D HEAD POSE & BINOCULAR EYE VECTOR ---
        # Measure socket width and eye aspect ratio (EAR) for both eyes
        l_width = np.hypot(lm[L_EYE_INNER].x - lm[L_EYE_OUTER].x, lm[L_EYE_INNER].y - lm[L_EYE_OUTER].y) + 1e-6
        l_height = np.hypot(lm[L_EYE_TOP].x - lm[L_EYE_BOTTOM].x, lm[L_EYE_TOP].y - lm[L_EYE_BOTTOM].y) + 1e-6
        r_width = np.hypot(lm[R_EYE_INNER].x - lm[R_EYE_OUTER].x, lm[R_EYE_INNER].y - lm[R_EYE_OUTER].y) + 1e-6
        r_height = np.hypot(lm[R_EYE_TOP].x - lm[R_EYE_BOTTOM].x, lm[R_EYE_TOP].y - lm[R_EYE_BOTTOM].y) + 1e-6

        l_ear = l_height / l_width
        r_ear = r_height / r_width
        ear = (l_ear + r_ear) / 2.0

        # Looking down naturally narrows the eye opening; only real blinks drop below BLINK_THRESHOLD (0.20)
        is_blink = ear < BLINK_THRESHOLD

        # Solve 3D Skull Pose
        face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_3D_IDS], dtype=np.float64)
        focal_len = w * 0.72
        cam_matrix = np.array([[focal_len, 0, w / 2.0], [0, focal_len, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success_pnp, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        raw_x = raw_y = raw_z = 0.0
        rmat = np.eye(3)
        if success_pnp:
            raw_x = tvec[0, 0]
            raw_y = tvec[1, 0]
            raw_z = tvec[2, 0]
            rmat, _ = cv2.Rodrigues(rvec)

        # Eye corner midpoints
        l_mid_x = (lm[L_EYE_INNER].x + lm[L_EYE_OUTER].x) / 2.0
        l_mid_y = (lm[L_EYE_INNER].y + lm[L_EYE_OUTER].y) / 2.0
        r_mid_x = (lm[R_EYE_INNER].x + lm[R_EYE_OUTER].x) / 2.0
        r_mid_y = (lm[R_EYE_INNER].y + lm[R_EYE_OUTER].y) / 2.0

        if not is_blink:
            # Measure iris displacement relative to neutral canthus center
            l_dx = (lm[LEFT_IRIS].x - l_mid_x) / l_width
            l_dy = (lm[LEFT_IRIS].y - l_mid_y) / l_width - IRIS_NEUTRAL_Y_OFFSET
            r_dx = (lm[RIGHT_IRIS].x - r_mid_x) / r_width
            r_dy = (lm[RIGHT_IRIS].y - r_mid_y) / r_width - IRIS_NEUTRAL_Y_OFFSET

            # Binocular average prevents single-eye jitter or reflections from skewing gaze
            last_good_iris_x = (l_dx + r_dx) / 2.0
            last_good_iris_y = (l_dy + r_dy) / 2.0

        eye_dx = last_good_iris_x
        eye_dy = last_good_iris_y

        # --- STEP 2: FUSE HEAD + EYE VECTORS ---
        head_x = -rmat[0, 2]
        head_y = -rmat[1, 2]
        raw_gaze_x = head_x + (eye_dx * EYE_WEIGHT_X)
        raw_gaze_y = head_y + (eye_dy * EYE_WEIGHT_Y)

        # Difference Calibration: compute displacement from neutral center
        dx = raw_gaze_x - calib_center[0]
        dy = raw_gaze_y - calib_center[1]

        # Quadrant routing: 0:TL (dx<0, dy<0), 1:TR (dx>=0, dy<0), 2:BR (dx>=0, dy>=0), 3:BL (dx<0, dy>=0)
        if dx < 0 and dy < 0:
            quad_idx = 0
        elif dx >= 0 and dy < 0:
            quad_idx = 1
        elif dx >= 0 and dy >= 0:
            quad_idx = 2
        else:
            quad_idx = 3

        sx, sy = scale_factors[quad_idx] if is_calibrated else (1.0, 1.0)
        eff_dx = dx * sx
        eff_dy = dy * sy

        left_span, right_span, top_span, bottom_span = full_spans
        span_x = left_span if eff_dx < 0 else right_span
        span_y = top_span if eff_dy < 0 else bottom_span

        # Map to normalized coordinates (0.50 is calibrated center)
        norm_x = 0.50 + 0.50 * (eff_dx / (span_x + 1e-6))
        norm_y = 0.50 + 0.50 * (eff_dy / (span_y + 1e-6))

        # Allow normalized gaze to extend past [0, 1] so cursor can glide beyond physical edges into 120% logical screen
        norm_x = np.clip(norm_x, -0.25, 1.25)
        norm_y = np.clip(norm_y, -0.25, 1.25)

        # Map to 120% logical screen centered on the physical display
        target_x = norm_x * LOGICAL_W - LOGICAL_OFFSET_X
        target_y = norm_y * LOGICAL_H - LOGICAL_OFFSET_Y

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

        # Clamp only when dispatching the physical mouse movement
        clamped_x = int(np.clip(cursor_x, 0, SCREEN_W - 1))
        clamped_y = int(np.clip(cursor_y, 0, SCREEN_H - 1))
        if clamped_x == 0 and clamped_y == 0:
            clamped_x, clamped_y = 1, 1  # Avoid PyAutoGUI FailSafeException at (0,0)

        if tracking_active:
            pg.moveTo(clamped_x, clamped_y)

        # Diagnostic flags: check if cursor hits screen edges or blink state
        stuck_tags = []
        if clamped_x == 0: stuck_tags.append("CLAMP_LEFT")
        elif clamped_x == SCREEN_W - 1: stuck_tags.append("CLAMP_RIGHT")
        if clamped_y == 0: stuck_tags.append("CLAMP_TOP")
        elif clamped_y == SCREEN_H - 1: stuck_tags.append("CLAMP_BOTTOM")
        if is_blink: stuck_tags.append("BLINK")

        # Telemetry logging to console log file
        frame_count += 1
        status_str = "ACTIVE" if tracking_active else "PAUSED"
        tag_str = f" [{' '.join(stuck_tags)}]" if stuck_tags else ""

        if frame_count % 5 == 0 or (stuck_tags and frame_count % 2 == 0):
            log_line = (
                f"[{time.strftime('%H:%M:%S')}] #{frame_count:05d} "
                f"EAR:{ear:.2f} | "
                f"Head:({head_x:+.2f},{head_y:+.2f}) | "
                f"Eye:({eye_dx:+.3f},{eye_dy:+.3f}) | "
                f"RawGaze:({raw_gaze_x:+.3f},{raw_gaze_y:+.3f}) | "
                f"Norm:({norm_x:.2f},{norm_y:.2f}) | "
                f"Cursor:({cursor_x:.0f},{cursor_y:.0f})->Move:({clamped_x},{clamped_y}) "
                f"[{status_str}]{tag_str}\n"
            )
            log_file.write(log_line)

        # --- STEP 4: DIAGNOSTIC HUD & VISUALIZATION ---
        status_color = (0, 255, 0) if tracking_active else (0, 165, 255)
        status_text = "ACTIVE (SPACE: pause)" if tracking_active else "PAUSED (SPACE: track | C: calib | R: reset)"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        hud_ear = f"EAR: {ear:.2f} ({'BLINK' if is_blink else 'OPEN'}) | Eye: dx={eye_dx:+.2f}, dy={eye_dy:+.2f}"
        cv2.putText(frame, hud_ear, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        hud_head = f"Head: pitch={head_y:+.2f}, yaw={head_x:+.2f} | RawGaze: ({raw_gaze_x:+.2f}, {raw_gaze_y:+.2f})"
        cv2.putText(frame, hud_head, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        stuck_badge = f" [STUCK: {' '.join(stuck_tags)}]" if stuck_tags else " [OK]"
        badge_color = (0, 0, 255) if stuck_tags else (0, 255, 0)
        hud_cursor = f"Norm: ({norm_x:.2f}, {norm_y:.2f}) -> Pos: ({clamped_x}, {clamped_y}){stuck_badge}"
        cv2.putText(frame, hud_cursor, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2)

        # Calibration HUD indicators & bullseye
        if not is_calibrated and calib_step < len(CALIB_STEPS):
            step_name, step_desc, (tx, ty) = CALIB_STEPS[calib_step]
            target_px = int(tx * w)
            target_py = int(ty * h)

            # Draw target bullseye on camera feed
            cv2.circle(frame, (target_px, target_py), 22, (0, 255, 255), 2)
            cv2.circle(frame, (target_px, target_py), 6, (0, 0, 255), -1)
            cv2.putText(frame, step_name, (target_px - 28, target_py - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Highlight banner at bottom of window
            cv2.rectangle(frame, (10, h - 45), (w - 10, h - 8), (20, 20, 20), -1)
            cv2.putText(frame, step_desc, (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        else:
            avg_sx = np.mean([sf[0] for sf in scale_factors])
            avg_sy = np.mean([sf[1] for sf in scale_factors])
            calib_str = f"CALIBRATED | Scale X:{avg_sx:.2f}x Y:{avg_sy:.2f}x | 'R' to reset"
            cv2.putText(frame, calib_str, (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

        # Draw moving gaze bubble and cursor on camera feed
        bx = int((bubble_x / SCREEN_W) * w)
        by = int((bubble_y / SCREEN_H) * h)
        cx = int((cursor_x / SCREEN_W) * w)
        cy = int((cursor_y / SCREEN_H) * h)
        hud_radius = max(6, int((bubble_radius / SCREEN_W) * w))
        cv2.circle(frame, (bx, by), hud_radius, (255, 255, 0), 2)  # Cyan gaze bubble
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)             # Red cursor point

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        tracking_active = not tracking_active
        msg = f"Tracking toggled: {'ACTIVE' if tracking_active else 'PAUSED'}"
        print(msg)
        log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    elif key == ord("r"):
        calib_step = 0
        is_calibrated = False
        calib_center = np.array([0.0, -0.14])
        scale_factors = [(1.75, 1.75), (1.75, 1.75), (1.75, 1.75), (1.75, 1.75)]
        full_spans = [0.35, 0.35, 0.24, 0.24]
        comfort_vectors = [np.array([-0.20, -0.15]), np.array([0.20, -0.15]), np.array([0.20, 0.15]), np.array([-0.20, 0.15])]
        full_vectors = [np.array([-0.35, -0.24]), np.array([0.35, -0.24]), np.array([0.35, 0.24]), np.array([-0.35, 0.24])]
        msg = "Reset calibration to defaults. Press 'C' to start 9-step calibration."
        print(msg)
        log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    elif key == ord("c"):
        now = time.perf_counter()
        if now - last_calib_time > 0.4:
            last_calib_time = now
            if calib_step == 0:
                # Step 0: Center Lock
                calib_center = np.array([raw_gaze_x, raw_gaze_y])
                msg = f"[CALIB 1/9] CENTER locked: ({calib_center[0]:.3f}, {calib_center[1]:.3f})"
                print(msg)
                log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                calib_step += 1
            elif 1 <= calib_step <= 4:
                # Steps 1-4: Comfortable reach (TL, TR, BR, BL)
                c_idx = calib_step - 1
                corner_name = CORNER_NAMES[c_idx]
                vec = np.array([raw_gaze_x, raw_gaze_y]) - calib_center
                comfort_vectors[c_idx] = vec
                msg = f"[CALIB {calib_step+1}/9] COMFORT {corner_name} recorded: dx={vec[0]:+.3f}, dy={vec[1]:+.3f}"
                print(msg)
                log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                calib_step += 1
            elif 5 <= calib_step <= 8:
                # Steps 5-8: Full reach (TL, TR, BR, BL)
                f_idx = calib_step - 5
                corner_name = CORNER_NAMES[f_idx]
                vec = np.array([raw_gaze_x, raw_gaze_y]) - calib_center
                full_vectors[f_idx] = vec
                msg = f"[CALIB {calib_step+1}/9] FULL {corner_name} recorded: dx={vec[0]:+.3f}, dy={vec[1]:+.3f}"
                print(msg)
                log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                calib_step += 1

                if calib_step == 9:
                    # Difference Scale calculation: abs(full) / abs(comfort)
                    new_scales = []
                    for i in range(4):
                        c_dx, c_dy = comfort_vectors[i]
                        f_dx, f_dy = full_vectors[i]
                        sx = np.clip(abs(f_dx) / max(abs(c_dx), 0.02), 1.0, 5.0)
                        sy = np.clip(abs(f_dy) / max(abs(c_dy), 0.02), 1.0, 5.0)
                        new_scales.append((float(sx), float(sy)))
                    scale_factors = new_scales

                    left_s = max(abs(full_vectors[0][0]), abs(full_vectors[3][0]), 0.15)
                    right_s = max(abs(full_vectors[1][0]), abs(full_vectors[2][0]), 0.15)
                    top_s = max(abs(full_vectors[0][1]), abs(full_vectors[1][1]), 0.10)
                    bottom_s = max(abs(full_vectors[2][1]), abs(full_vectors[3][1]), 0.10)
                    full_spans = [left_s, right_s, top_s, bottom_s]
                    is_calibrated = True

                    summary_msg = (
                        f"=== CALIBRATION COMPLETE ===\n"
                        f"Center: ({calib_center[0]:.3f}, {calib_center[1]:.3f})\n"
                        f"Scale Factors: TL=({scale_factors[0][0]:.2f}, {scale_factors[0][1]:.2f}) | "
                        f"TR=({scale_factors[1][0]:.2f}, {scale_factors[1][1]:.2f}) | "
                        f"BR=({scale_factors[2][0]:.2f}, {scale_factors[2][1]:.2f}) | "
                        f"BL=({scale_factors[3][0]:.2f}, {scale_factors[3][1]:.2f})\n"
                        f"Full Spans: L={left_s:.3f}, R={right_s:.3f}, T={top_s:.3f}, B={bottom_s:.3f}"
                    )
                    print(summary_msg)
                    log_file.write(f"[{time.strftime('%H:%M:%S')}] {summary_msg}\n")
            else:
                msg = "Already calibrated. Press 'R' to reset and recalibrate."
                print(msg)

cap.release()
cv2.destroyAllWindows()
log_file.write(f"[{time.strftime('%H:%M:%S')}] Session Ended\n")
log_file.close()