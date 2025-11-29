import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
from collections import deque
import base64
import json

class KalmanFilter:
    def __init__(self):
        # Enhanced state vector: [x, y, dx, dy, w, h, dw, dh]
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurements
        
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ], np.float32)
        
        # Enhanced transition matrix for better size and velocity prediction
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Tuned noise parameters
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        self.initialized = False
        
    def init(self, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        self.kalman.statePre = np.array([[x], [y], [0], [0], [w], [h], [0], [0]], np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()
        self.initialized = True
        
    def predict(self) -> Tuple[int, int, int, int]:
        if not self.initialized:
            return None
            
        prediction = self.kalman.predict()
        x, y, _, _, w, h, _, _ = prediction.flatten()
        return (int(x), int(y), int(w), int(h))
        
    def correct(self, bbox: Tuple[int, int, int, int]):
        if not self.initialized:
            self.init(bbox)
            return bbox
            
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], np.float32)
        corrected = self.kalman.correct(measurement)
        x, y, _, _, w, h, _, _ = corrected.flatten()
        return (int(x), int(y), int(w), int(h))

class MovementAnalyzer:
    def __init__(self, collision_th: float):
        # Existing movement tracking attributes
        self.position_history = deque(maxlen=15)
        self.prev_position = None
        self.base_height = None
        self.ground_level = None
        self.consecutive_up_frames = 0
        self.consecutive_down_frames = 0
        self.in_air = False
        self.tackle_cooldown = 0
        self.last_movement = None
        self.movement_start_y = None
        self.current_frame = 0
        self.jump_start_frame = 0
        self.requires_landing = False
        self.landing_timeout = 45
        self.last_tackle_direction = None
        self.movement_lockout = 60
        self.last_movement_frame = 0
        self.initial_y = None
        self.ground_y = None
        self.jump_detected = False
        
        # New collision detection parameters
        self.collision_cooldown = 0
        self.collision_frames = 0
        self.contact_threshold = collision_th  # Minimum overlap ratio for collision
        self.min_collision_frames = 2
        self.potential_collision = False
        self.contact_frames = 0
        
    def check_contact(self, other_bbox):
        """Check if there is significant contact between player and another bbox"""
        if not self.prev_position:
            return False
            
        x1 = self.prev_position['x']
        y1 = self.prev_position['y']
        w1 = self.prev_position['w']
        h1 = self.prev_position['h']
        
        x2, y2, w2, h2 = other_bbox
        
        # Calculate the intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        # Calculate overlap area
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        min_area = min(w1 * h1, w2 * h2)
        overlap_ratio = overlap_area / min_area
        print(overlap_ratio)
        
        return overlap_ratio > self.contact_threshold

    def update(self, bbox: Tuple[int, int, int, int], other_players: List[Tuple[int, int, int, int]] = None) -> Optional[str]:
        self.current_frame += 1
        x, y, w, h = [float(val) for val in bbox]
        current_pos = {'x': x, 'y': y, 'w': w, 'h': h}
        
        if self.prev_position is None:
            self.prev_position = current_pos
            self.initial_y = y
            self.ground_y = y + h
            return None
            
        y_change = current_pos['y'] - self.prev_position['y']
        x_change = current_pos['x'] - self.prev_position['x']
        movement = None

        # Jump detection
        if not self.requires_landing:
            if y_change < -12:
                self.consecutive_up_frames += 1
                if self.consecutive_up_frames >= 3:
                    if len(self.position_history) >= 3:
                        recent_positions = list(self.position_history)[-3:]
                        y_velocities = [recent_positions[i]['y'] - recent_positions[i-1]['y'] 
                                      for i in range(1, len(recent_positions))]
                        
                        if all(v < -8 for v in y_velocities):
                            total_y_change = current_pos['y'] - self.ground_y
                            if abs(total_y_change) > 25:
                                movement = "JUMPING"
                                self.requires_landing = True
                                self.jump_start_frame = self.current_frame
                                self.in_air = True
            else:
                self.consecutive_up_frames = 0

        # Landing detection
        elif self.requires_landing:
            if y_change > 8:
                self.consecutive_down_frames += 1
                if self.consecutive_down_frames >= 3:
                    if abs(current_pos['y'] + current_pos['h'] - self.ground_y) < 25:
                        movement = "LANDING"
                        self.requires_landing = False
                        self.consecutive_down_frames = 0
                        self.in_air = False
            else:
                self.consecutive_down_frames = 0
                
            if self.current_frame - self.jump_start_frame > 30:
                self.requires_landing = False
                self.in_air = False

        # Tackle detection
        if not self.in_air and abs(x_change) > 15 and abs(y_change) < 10:
            if self.tackle_cooldown == 0:
                movement = "TACKLE"
                self.tackle_cooldown = 45
        if self.tackle_cooldown > 0:
            self.tackle_cooldown -= 1

        # Enhanced collision detection using physical contact
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        elif other_players:
            for other_bbox in other_players:
                if self.check_contact(other_bbox):
                    self.contact_frames += 1
                    if self.contact_frames >= self.min_collision_frames:
                        movement = "COLLISION"
                        self.collision_cooldown = 30
                        self.contact_frames = 0
                        break
                else:
                    self.contact_frames = max(0, self.contact_frames - 1)
        
        self.position_history.append(current_pos)
        self.prev_position = current_pos
        
        return movement if movement != self.last_movement else None

class PlayerTracker:
    def __init__(self, collision_th: float):
        self.tracked_player = None
        self.tracking_initialized = False
        self.tracker = cv2.TrackerCSRT_create()
        self.kalman = KalmanFilter()
        self.movement_analyzer = MovementAnalyzer(collision_th)
        self.frames_since_detection = 0
        self.max_frames_lost = 45
        self.last_valid_size = None
        self.template = None
        self.template_history = deque(maxlen=5)
        self.template_update_interval = 15
        self.last_reliable_bbox = None
        
    def init_tracking(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.kalman.init(bbox)
        self.tracking_initialized = True
        self.tracked_player = bbox
        self.frames_since_detection = 0
        self.last_valid_size = (bbox[2], bbox[3])
        self.last_reliable_bbox = bbox
        
        # Initialize template history
        x, y, w, h = [int(v) for v in bbox]
        initial_template = frame[y:y+h, x:x+w].copy()
        for _ in range(5):
            self.template_history.append(initial_template)
        self.template = initial_template
        
    def _extract_features(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return hsv, edges
        
    def _compare_patches(self, current_patch: np.ndarray, template: np.ndarray, 
                        size_threshold: float = 0.4) -> float:
        if current_patch.size == 0 or template.size == 0:
            return 0.0
            
        h1, w1 = current_patch.shape[:2]
        h2, w2 = template.shape[:2]
        size_diff = abs(1 - (h1 * w1) / (h2 * w2))
        if size_diff > size_threshold:
            return 0.0
            
        try:
            template_resized = cv2.resize(template, (current_patch.shape[1], current_patch.shape[0]))
            
            current_hsv, current_edges = self._extract_features(current_patch)
            template_hsv, template_edges = self._extract_features(template_resized)
            
            hist_current = cv2.calcHist([current_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist_template = cv2.calcHist([template_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
            color_similarity = cv2.compareHist(hist_current, hist_template, cv2.HISTCMP_CORREL)
            
            edge_similarity = cv2.matchTemplate(current_edges, template_edges, cv2.TM_CCOEFF_NORMED)[0][0]
            
            return 0.7 * color_similarity + 0.3 * max(0, edge_similarity)
        except cv2.error:
            return 0.0
            
    def _validate_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> bool:
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False
            
        if w < 10 or h < 10 or w > frame_w/2 or h > frame_h/2:
            return False
            
        aspect_ratio = w/h
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return False
            
        return True
        
    def update(self, frame: np.ndarray):
        if not self.tracking_initialized:
            return False, None

        # Kalman prediction (may be None before first init)
        predicted_bbox = self.kalman.predict()
        try:
            success, tracked_bbox = self.tracker.update(frame)
        except cv2.error:
            success, tracked_bbox = False, None

        # Try OpenCV tracker first
        success, tracked_bbox = self.tracker.update(frame)

        # --- 1) Normal tracker success path ---
        if success and self._validate_bbox(tracked_bbox, frame.shape):
            x, y, w, h = [int(v) for v in tracked_bbox]

            # Reject boxes that are way smaller/larger than the usual player (e.g. ball)
            if self.last_valid_size is not None:
                lw, lh = self.last_valid_size
                area_last = lw * lh
                area_now = w * h
                ratio = area_now / max(area_last, 1)

                # If it shrinks a lot or explodes in size, treat as tracking failure
                if ratio < 0.6 or ratio > 1.4:
                    success = False

            if success:
                current_patch = frame[y:y + h, x:x + w]

                # Optional: template similarity & update (your existing logic)
                if current_patch.size != 0 and len(self.template_history) > 0:
                    similarities = [
                        self._compare_patches(current_patch, template)
                        for template in self.template_history
                    ]
                    max_similarity = max(similarities)

                    if max_similarity > 0.5:
                        if self.frames_since_detection % self.template_update_interval == 0:
                            self.template_history.append(current_patch.copy())
                            self.template = current_patch.copy()

                # Kalman correction + bookkeeping (Kalman stays!)
                corrected_bbox = self.kalman.correct((x, y, w, h))
                cx, cy, cw, ch = corrected_bbox
                corrected_bbox = (int(cx), int(cy), int(cw), int(ch))

                self.tracked_player = corrected_bbox
                self.frames_since_detection = 0
                self.last_valid_size = (corrected_bbox[2], corrected_bbox[3])
                self.last_reliable_bbox = corrected_bbox

                movement = self.movement_analyzer.update(corrected_bbox)
                if movement:
                    return True, corrected_bbox, movement
                return True, corrected_bbox

        # --- 2) Tracker failed this frame: use Kalman prediction + your snippet ---
        self.frames_since_detection += 1

        if (
            self.frames_since_detection <= self.max_frames_lost
            and predicted_bbox is not None
        ):
            px, py, pw, ph = predicted_bbox

            # Constrain predicted size around last reliable bbox if we have one
            if self.last_reliable_bbox is not None:
                lw, lh = self.last_reliable_bbox[2], self.last_reliable_bbox[3]
                pw = int(min(max(pw, lw * 0.8), lw * 1.2))
                ph = int(min(max(ph, lh * 0.8), lh * 1.2))

            # Clamp into frame bounds so tracker.init doesn't crash
            frame_h, frame_w = frame.shape[:2]
            px = max(0, min(int(px), frame_w - 1))
            py = max(0, min(int(py), frame_h - 1))
            pw = max(1, min(int(pw), frame_w - px))
            ph = max(1, min(int(ph), frame_h - py))

            predicted_bbox = (int(px), int(py), int(pw), int(ph))
            if predicted_bbox[2] <= 0 or predicted_bbox[3] <= 0:
                self.tracking_initialized = False
                return False, None

            # Use Kalman prediction (your snippet, with safety)
            self.tracked_player = predicted_bbox

            movement = self.movement_analyzer.update(predicted_bbox)
            if movement:
                return True, predicted_bbox, movement
            return True, predicted_bbox

        # --- 3) No prediction available: give up after too many frames ---
        if self.frames_since_detection > self.max_frames_lost:
            self.tracking_initialized = False
            return False, None

        # During grace period but no good bbox: keep last known or mark as lost
        return False, self.tracked_player

def detect_movement(player_positions: deque,  jump_th: int, landing_th: int, tackle_th: int, threshold_multiplier: float = 1.0,) -> Optional[str]:
    if len(player_positions) < 5:
        return None

    recent_positions = list(player_positions)[-5:]
    start_pos = recent_positions[0]
    end_pos = recent_positions[-1]

    vertical_movement = end_pos[1] - start_pos[1]
    horizontal_movement = end_pos[0] - start_pos[0]

    jump_threshold = -1 * jump_th * threshold_multiplier
    landing_threshold = landing_th * threshold_multiplier
    tackle_threshold = tackle_th * threshold_multiplier
    movement_threshold = 20 * threshold_multiplier
    print([vertical_movement,jump_threshold,landing_threshold,horizontal_movement,movement_threshold])

    if vertical_movement < jump_threshold and abs(horizontal_movement) < movement_threshold:
        return "JUMPING"
    elif vertical_movement > landing_threshold and abs(horizontal_movement) < movement_threshold:
        return "LANDING"
    elif abs(horizontal_movement) > tackle_threshold and abs(vertical_movement) < movement_threshold:
        return "TACKLE"

    return None

def format_risk_display(movement_counts: Dict[str, int], risk_score: int) -> Tuple[List[str], str, tuple]:
    # Base risk weights for each action
    weights = {
        "JUMPING": 1,
        "LANDING": 2,
        "TACKLE": 4,
        "COLLISION": 5
    }
    
    # Calculate individual scores
    scores = {
        move: count * weights[move] 
        for move, count in movement_counts.items()
    }
    
    # Format display strings
    display_lines = [
        f"JUMPING & LANDING: {movement_counts['LANDING'] + movement_counts['JUMPING']} (Score: {scores['LANDING'] + scores['JUMPING']})",
        f"TACKLE: {movement_counts['TACKLE']} (Score: {scores['TACKLE']})",
        f"COLLISION: {movement_counts['COLLISION']} (Score: {scores['COLLISION']})"
    ]
    
    # Determine risk level
    if risk_score < 20:
        risk_level = "Low"
        color = (0, 255, 0)  # Green
    elif risk_score < 40:
        risk_level = "Moderate"
        color = (0, 255, 255)  # Yellow
    else:
        risk_level = "High"
        color = (0, 0, 255)  # Red
        
    total_line = f"Total score: {risk_score} Risk: {risk_level}"
    
    return display_lines, total_line, color

def calculate_risk_score(movement_counts: Dict[str, int], frame_counter: int, time_weights: bool = True) -> int:
    # Fixed base risk weights
    base_risk_weights = {
        "JUMPING": 1,
        "LANDING": 2,
        "TACKLE": 4,
        "COLLISION": 5
    }
    print(movement_counts)
    
    if time_weights:
        total_actions = sum(movement_counts.values())
        if total_actions > 0:
            action_frequency = total_actions / frame_counter
            frequency_multiplier = min(2.0, max(1.0, action_frequency * 100))
            risk_weights = {k: v for k, v in base_risk_weights.items()}  # Remove frequency multiplier for jump/land
        else:
            risk_weights = base_risk_weights
    else:
        risk_weights = base_risk_weights
    
    # Calculate the score directly using the base weights
    score = (
        movement_counts["JUMPING"] * base_risk_weights["JUMPING"] +
        movement_counts["LANDING"] * base_risk_weights["LANDING"] +
        movement_counts["TACKLE"] * base_risk_weights["TACKLE"] +
        movement_counts["COLLISION"] * base_risk_weights["COLLISION"]
    )
    
    return score

def format_risk_display(movement_counts: Dict[str, int], risk_score: int) -> Tuple[List[str], str, tuple]:
    weights = {
        "JUMPING": 1,
        "LANDING": 2,
        "TACKLE": 4,
        "COLLISION": 5
    }
    
    # Calculate individual scores using fixed weights
    scores = {
        move: count * weights[move] 
        for move, count in movement_counts.items()
    }
    
    # Format display strings
    display_lines = [
        f"LANDING & JUMPING: {movement_counts['LANDING'] + movement_counts['JUMPING']} (Score: {scores['LANDING'] + scores['JUMPING']})",
        f"TACKLE: {movement_counts['TACKLE']} (Score: {scores['TACKLE']})",
        f"COLLISION: {movement_counts['COLLISION']} (Score: {scores['COLLISION']})"
    ]
    
    if risk_score < 20:
        risk_level = "Low"
        color = (0, 255, 0)
    elif risk_score < 40:
        risk_level = "Moderate"
        color = (0, 255, 255)
    else:
        risk_level = "High"
        color = (0, 0, 255)
        
    total_line = f"Total score: {risk_score} Risk: {risk_level}"
    
    return display_lines, total_line, color

def process_video(
    video_path: str,
    txt_output_path: str,
    main_bbox: Tuple[int, int, int, int],
    other_bboxes: List[Tuple[int, int, int, int]],
):
    # 🔹 Fixed default thresholds
    jump_th = 15        # was default on slider
    landing_th = 15     # was default on slider
    tackle_th = 15      # was default on slider
    collision_th = 0.5  # was default on slider

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    # Read first frame to validate bboxes and to init trackers
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video frame")

    # --- validate and clean main_bbox ---
    x, y, w, h = map(int, main_bbox)
    frame_h, frame_w = frame.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise RuntimeError("Invalid main bbox from client")

    main_bbox = (x, y, w, h)

    # --- validate and clean other_bboxes ---
    cleaned_other_players = []
    for ob in other_bboxes:
        ox, oy, ow, oh = map(int, ob)
        if ow <= 0 or oh <= 0:
            continue
        if ox < 0 or oy < 0 or ox + ow > frame_w or oy + oh > frame_h:
            continue
        cleaned_other_players.append((ox, oy, ow, oh))
    other_players = cleaned_other_players

    # Reset video to start (so processing loop starts at frame 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- VideoWriter setup ---
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or np.isnan(fps):
        fps = 30.0  # fallback so video is not 0:00

        # --- Use MJPEG in AVI (usually playable in browsers) ---
    output_path = tempfile.mktemp(suffix=".avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter with MJPG/AVI.")
    
    print(f"[VideoWriter] Using MJPG in AVI, fps: {fps}")


    if out is None or not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter – cannot create output video.")

    # --- Initialize trackers ---
    player_tracker = PlayerTracker(collision_th)
    other_trackers = [PlayerTracker(collision_th) for _ in other_players]

    # Use the first frame we already read to init trackers
    player_tracker.init_tracking(frame, main_bbox)
    for tracker, bbox in zip(other_trackers, other_players):
        tracker.init_tracking(frame, bbox)

    # Tracking state
    player_positions = deque(maxlen=15)
    last_movement = None
    movement_counts = {"JUMPING": 0, "LANDING": 0, "TACKLE": 0, "COLLISION": 0}
    frame_counter = 0

    # --- Main processing loop ---
    with open(txt_output_path, "w") as txt_file:
        # We already used the first frame; continue from current position
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            if player_tracker.tracking_initialized:
                # Track other players first
                other_bboxes_frame = []
                for idx, tracker in enumerate(other_trackers):
                    if tracker.tracking_initialized:
                        res = tracker.update(frame)
                        success, bbox = res[:2]
                        if success:
                            other_bboxes_frame.append(bbox)
                            ox, oy, ow, oh = [int(v) for v in bbox]
                            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"Player {idx + 1}",
                                (ox, oy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                # Update main player tracking with other players' positions
                tracking_result = player_tracker.update(frame)
                if len(tracking_result) == 3:
                    success, bbox, movement = tracking_result
                else:
                    success, bbox = tracking_result
                    movement = None

                if success and bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Main Player",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                    player_tracker.tracked_player = (x, y, w, h)
                    player_center = (x + w // 2, y + h // 2)
                    player_positions.append(player_center)

                    # YOLO-style normalized coordinates
                    center_x = (x + w / 2) / frame_width
                    center_y = (y + h / 2) / frame_height
                    norm_width = w / frame_width
                    norm_height = h / frame_height

                    # Movement using current frame's other boxes
                    movement = player_tracker.movement_analyzer.update(
                        (x, y, w, h), other_bboxes_frame
                    )

                    # If tracker didn't signal, use position-based detection
                    if movement is None:
                        threshold_multiplier = (
                            player_tracker.last_valid_size[1] / 100.0
                            if player_tracker.last_valid_size
                            else 1.0
                        )
                        movement = detect_movement(
                            player_positions,
                            jump_th,
                            landing_th,
                            tackle_th,
                            threshold_multiplier,
                        )

                    if movement and movement != last_movement:
                        movement_counts[movement] += 1
                        last_movement = movement

                    # Risk metrics overlay
                    risk_score = calculate_risk_score(movement_counts, frame_counter)
                    display_lines, total_line, risk_color = format_risk_display(
                        movement_counts, risk_score
                    )

                    # Write label line
                    txt_file.write(
                        f"{0 if movement is None else MOVEMENT_MAPPING[movement]} "
                        f"{center_x:.6f} {center_y:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}\n"
                    )

                    # Draw text overlays
                    y_offset = 30
                    for line in display_lines:
                        cv2.putText(
                            frame,
                            line,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        y_offset += 25

                    cv2.putText(
                        frame,
                        total_line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        risk_color,
                        2,
                    )

            out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path, txt_output_path

# Gradio Interface
# with gr.Blocks() as app2:
    gr.Markdown("# Soccer Player Tracking and Risk Analysis with Coordinate Exports")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Soccer Video")
            txt_output = gr.Textbox(label="Output TXT Path", placeholder="output_labels.txt", value="output_labels.txt")
            track_btn = gr.Button("Track Player, Analyze Risk, and Export Labels")
        
        with gr.Column():
            output_video = gr.Video(label="Processed Video", format="mp4")
            output_txt = gr.File(label="Generated Labels (YOLO format)")
            output_txt_preview = gr.Textbox(
            label="Labels Preview",
            lines=10,
            interactive=True
    )


    track_btn.click(
        fn=process_video,
        inputs=[video_input, txt_output],
        outputs=[output_video, output_txt]
    )

MOVEMENT_MAPPING = {
    "JUMPING": 1, 
    "LANDING": 2, 
    "TACKLE": 1, 
    "COLLISION": 1
}

#if __name__ == "__main__":
    #try:
        #app2.launch(
            #share=True, 
            #debug=True, 
            #show_error=True
        #)
    #except Exception as e:
       # print(f"Detailed Error: {e}")
