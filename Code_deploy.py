#import os
#os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import tempfile
from typing import List, Tuple, Dict, Optional
from collections import deque
import json  # kept in case you use it elsewhere


# ---------------------------------------------------------
# Helper: CSRT tracker creation that works on different builds
# ---------------------------------------------------------
def create_tracker():
    """
    Try to create the best available OpenCV tracker.
    Preference: CSRT -> KCF -> MOSSE -> MIL.
    Works with both new (cv2.TrackerX_create) and legacy (cv2.legacy.TrackerX_create) APIs.
    """
    # --- CSRT ---
    if hasattr(cv2, "TrackerCSRT_create"):
        print("[Tracker] Using TrackerCSRT")
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        print("[Tracker] Using legacy.TrackerCSRT")
        return cv2.legacy.TrackerCSRT_create()

    # --- KCF ---
    if hasattr(cv2, "TrackerKCF_create"):
        print("[Tracker] Using TrackerKCF")
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        print("[Tracker] Using legacy.TrackerKCF")
        return cv2.legacy.TrackerKCF_create()

    # --- MOSSE ---
    if hasattr(cv2, "TrackerMOSSE_create"):
        print("[Tracker] Using TrackerMOSSE")
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        print("[Tracker] Using legacy.TrackerMOSSE")
        return cv2.legacy.TrackerMOSSE_create()

    # --- MIL as last fallback ---
    if hasattr(cv2, "TrackerMIL_create"):
        print("[Tracker] Using TrackerMIL")
        return cv2.TrackerMIL_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        print("[Tracker] Using legacy.TrackerMIL")
        return cv2.legacy.TrackerMIL_create()

    # If we get here, this OpenCV build has no usable tracker
    raise RuntimeError(
        "No supported OpenCV tracker (CSRT/KCF/MOSSE/MIL) found in this build."
    )

# ---------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------
class KalmanFilter:
    def __init__(self):
        # Enhanced state vector: [x, y, dx, dy, w, h, dw, dh]
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurements

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ],
            np.float32,
        )

        # Enhanced transition matrix for better size and velocity prediction
        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        # Tuned noise parameters
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        self.initialized = False

    def init(self, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        self.kalman.statePre = np.array(
            [[x], [y], [0], [0], [w], [h], [0], [0]], np.float32
        )
        self.kalman.statePost = self.kalman.statePre.copy()
        self.initialized = True

    def predict(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.initialized:
            return None

        prediction = self.kalman.predict()
        x, y, _, _, w, h, _, _ = prediction.flatten()
        return int(x), int(y), int(w), int(h)

    def correct(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if not self.initialized:
            self.init(bbox)
            return bbox

        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], np.float32)
        corrected = self.kalman.correct(measurement)
        x, y, _, _, w, h, _, _ = corrected.flatten()
        return int(x), int(y), int(w), int(h)


# ---------------------------------------------------------
# Movement Analyzer
# ---------------------------------------------------------
class MovementAnalyzer:
    def __init__(self, collision_th: float):
        # Movement tracking attributes
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

        # Collision detection parameters
        self.collision_cooldown = 0
        self.collision_frames = 0
        self.contact_threshold = collision_th  # Minimum overlap ratio
        self.min_collision_frames = 2
        self.potential_collision = False
        self.contact_frames = 0

    def check_contact(self, other_bbox) -> bool:
        """Check if there is significant contact between player and another bbox"""
        if not self.prev_position:
            return False

        x1 = self.prev_position["x"]
        y1 = self.prev_position["y"]
        w1 = self.prev_position["w"]
        h1 = self.prev_position["h"]

        x2, y2, w2, h2 = other_bbox

        # Intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return False

        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        min_area = min(w1 * h1, w2 * h2)
        if min_area <= 0:
            return False

        overlap_ratio = overlap_area / min_area
        print(overlap_ratio)

        return overlap_ratio > self.contact_threshold

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        other_players: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Optional[str]:
        self.current_frame += 1
        x, y, w, h = [float(val) for val in bbox]
        current_pos = {"x": x, "y": y, "w": w, "h": h}

        if self.prev_position is None:
            self.prev_position = current_pos
            self.initial_y = y
            self.ground_y = y + h
            return None

        y_change = current_pos["y"] - self.prev_position["y"]
        x_change = current_pos["x"] - self.prev_position["x"]
        movement = None

        # Jump detection
        if not self.requires_landing:
            if y_change < -12:
                self.consecutive_up_frames += 1
                if self.consecutive_up_frames >= 3:
                    if len(self.position_history) >= 3:
                        recent_positions = list(self.position_history)[-3:]
                        y_velocities = [
                            recent_positions[i]["y"] - recent_positions[i - 1]["y"]
                            for i in range(1, len(recent_positions))
                        ]

                        if all(v < -8 for v in y_velocities):
                            total_y_change = current_pos["y"] - self.ground_y
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
                    if abs(current_pos["y"] + current_pos["h"] - self.ground_y) < 25:
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

        # Collision detection
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

        if movement and movement != self.last_movement:
            self.last_movement = movement
            return movement

        return None


# ---------------------------------------------------------
# Player Tracker
# ---------------------------------------------------------
class PlayerTracker:
    def __init__(self, collision_th: float):
        self.tracked_player = None
        self.tracking_initialized = False
        self.tracker = create_tracker()
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
        # Always use helper to support different OpenCV builds
        self.tracker = create_tracker()
        self.tracker.init(frame, bbox)
        self.kalman.init(bbox)
        self.tracking_initialized = True
        self.tracked_player = bbox
        self.frames_since_detection = 0
        self.last_valid_size = (bbox[2], bbox[3])
        self.last_reliable_bbox = bbox

        # Initialize template history
        x, y, w, h = [int(v) for v in bbox]
        initial_template = frame[y : y + h, x : x + w].copy()
        for _ in range(5):
            self.template_history.append(initial_template)
        self.template = initial_template

    def _extract_features(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return hsv, edges

    def _compare_patches(
        self, current_patch: np.ndarray, template: np.ndarray, size_threshold: float = 0.4
    ) -> float:
        if current_patch.size == 0 or template.size == 0:
            return 0.0

        h1, w1 = current_patch.shape[:2]
        h2, w2 = template.shape[:2]
        size_diff = abs(1 - (h1 * w1) / (h2 * w2))
        if size_diff > size_threshold:
            return 0.0

        try:
            template_resized = cv2.resize(
                template, (current_patch.shape[1], current_patch.shape[0])
            )

            current_hsv, current_edges = self._extract_features(current_patch)
            template_hsv, template_edges = self._extract_features(template_resized)

            hist_current = cv2.calcHist(
                [current_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
            )
            hist_template = cv2.calcHist(
                [template_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
            )
            cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
            color_similarity = cv2.compareHist(
                hist_current, hist_template, cv2.HISTCMP_CORREL
            )

            edge_similarity = cv2.matchTemplate(
                current_edges, template_edges, cv2.TM_CCOEFF_NORMED
            )[0][0]

            return 0.7 * color_similarity + 0.3 * max(0, edge_similarity)
        except cv2.error:
            return 0.0

    def _validate_bbox(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> bool:
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False

        if w < 10 or h < 10 or w > frame_w / 2 or h > frame_h / 2:
            return False

        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return False

        return True

    def update(self, frame: np.ndarray):
        if not self.tracking_initialized:
            return False, None

        predicted_bbox = self.kalman.predict()

        try:
            success, tracked_bbox = self.tracker.update(frame)
        except cv2.error:
            success, tracked_bbox = False, None

        # 1) Normal tracker success path
        if success and tracked_bbox is not None and self._validate_bbox(
            tracked_bbox, frame.shape
        ):
            x, y, w, h = [int(v) for v in tracked_bbox]

            # Reject boxes that are way smaller/larger than last valid size
            if self.last_valid_size is not None:
                lw, lh = self.last_valid_size
                area_last = lw * lh
                area_now = w * h
                ratio = area_now / max(area_last, 1)
                if ratio < 0.6 or ratio > 1.4:
                    success = False

            if success:
                current_patch = frame[y : y + h, x : x + w]

                if current_patch.size != 0 and len(self.template_history) > 0:
                    similarities = [
                        self._compare_patches(current_patch, template)
                        for template in self.template_history
                    ]
                    max_similarity = max(similarities)

                    if max_similarity > 0.5:
                        if (
                            self.frames_since_detection
                            % self.template_update_interval
                            == 0
                        ):
                            self.template_history.append(current_patch.copy())
                            self.template = current_patch.copy()

                corrected_bbox = self.kalman.correct((x, y, w, h))
                cx, cy, cw, ch = corrected_bbox
                corrected_bbox = int(cx), int(cy), int(cw), int(ch)

                self.tracked_player = corrected_bbox
                self.frames_since_detection = 0
                self.last_valid_size = (corrected_bbox[2], corrected_bbox[3])
                self.last_reliable_bbox = corrected_bbox

                movement = self.movement_analyzer.update(corrected_bbox)
                if movement:
                    return True, corrected_bbox, movement
                return True, corrected_bbox

        # 2) Tracker failed: rely on Kalman prediction if available
        self.frames_since_detection += 1

        if self.frames_since_detection <= self.max_frames_lost and predicted_bbox:
            px, py, pw, ph = predicted_bbox

            if self.last_reliable_bbox is not None:
                lw, lh = self.last_reliable_bbox[2], self.last_reliable_bbox[3]
                pw = int(min(max(pw, lw * 0.8), lw * 1.2))
                ph = int(min(max(ph, lh * 0.8), lh * 1.2))

            frame_h, frame_w = frame.shape[:2]
            px = max(0, min(int(px), frame_w - 1))
            py = max(0, min(int(py), frame_h - 1))
            pw = max(1, min(int(pw), frame_w - px))
            ph = max(1, min(int(ph), frame_h - py))

            predicted_bbox = int(px), int(py), int(pw), int(ph)
            if predicted_bbox[2] <= 0 or predicted_bbox[3] <= 0:
                self.tracking_initialized = False
                return False, None

            self.tracked_player = predicted_bbox

            movement = self.movement_analyzer.update(predicted_bbox)
            if movement:
                return True, predicted_bbox, movement
            return True, predicted_bbox

        # 3) Too many lost frames
        if self.frames_since_detection > self.max_frames_lost:
            self.tracking_initialized = False
            return False, None

        return False, self.tracked_player


# ---------------------------------------------------------
# Movement detection helpers
# ---------------------------------------------------------
def detect_movement(
    player_positions: deque,
    jump_th: int,
    landing_th: int,
    tackle_th: int,
    threshold_multiplier: float = 1.0,
) -> Optional[str]:
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
    print(
        [
            vertical_movement,
            jump_threshold,
            landing_threshold,
            horizontal_movement,
            movement_threshold,
        ]
    )

    if vertical_movement < jump_threshold and abs(horizontal_movement) < movement_threshold:
        return "JUMPING"
    elif vertical_movement > landing_threshold and abs(horizontal_movement) < movement_threshold:
        return "LANDING"
    elif abs(horizontal_movement) > tackle_threshold and abs(vertical_movement) < movement_threshold:
        return "TACKLE"

    return None


def calculate_risk_score(
    movement_counts: Dict[str, int], frame_counter: int, time_weights: bool = True
) -> int:
    base_risk_weights = {
        "JUMPING": 1,
        "LANDING": 2,
        "TACKLE": 4,
        "COLLISION": 5,
    }
    print(movement_counts)

    if time_weights:
        total_actions = sum(movement_counts.values())
        if total_actions > 0:
            action_frequency = total_actions / max(frame_counter, 1)
            _frequency_multiplier = min(2.0, max(1.0, action_frequency * 100))
            _ = _frequency_multiplier  # not used now, but kept for future tuning

    score = (
        movement_counts["JUMPING"] * base_risk_weights["JUMPING"]
        + movement_counts["LANDING"] * base_risk_weights["LANDING"]
        + movement_counts["TACKLE"] * base_risk_weights["TACKLE"]
        + movement_counts["COLLISION"] * base_risk_weights["COLLISION"]
    )
    return score


def format_risk_display(
    movement_counts: Dict[str, int], risk_score: int
) -> Tuple[List[str], str, tuple]:
    weights = {
        "JUMPING": 1,
        "LANDING": 2,
        "TACKLE": 4,
        "COLLISION": 5,
    }

    scores = {move: count * weights[move] for move, count in movement_counts.items()}

    display_lines = [
        f"LANDING & JUMPING: {movement_counts['LANDING'] + movement_counts['JUMPING']} "
        f"(Score: {scores['LANDING'] + scores['JUMPING']})",
        f"TACKLE: {movement_counts['TACKLE']} (Score: {scores['TACKLE']})",
        f"COLLISION: {movement_counts['COLLISION']} (Score: {scores['COLLISION']})",
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


# ---------------------------------------------------------
# Movement → YOLO label mapping
# ---------------------------------------------------------
MOVEMENT_MAPPING = {
    "JUMPING": 1,
    "LANDING": 2,
    "TACKLE": 1,
    "COLLISION": 1,
}


# ---------------------------------------------------------
# Main video processing function (used by Flask API)
# ---------------------------------------------------------
def process_video(
    video_path: str,
    txt_output_path: str,
    main_bbox: Tuple[int, int, int, int],
    other_bboxes: List[Tuple[int, int, int, int]],
):
    # Fixed thresholds
    jump_th = 15
    landing_th = 15
    tackle_th = 15
    collision_th = 0.5

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    # Read first frame to validate bboxes and init trackers
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video frame")

    # Validate main bbox
    x, y, w, h = map(int, main_bbox)
    frame_h, frame_w = frame.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise RuntimeError("Invalid main bbox from client")
    main_bbox = (x, y, w, h)

    # Validate other bboxes
    cleaned_other_players = []
    for ob in other_bboxes:
        ox, oy, ow, oh = map(int, ob)
        if ow <= 0 or oh <= 0:
            continue
        if ox < 0 or oy < 0 or ox + ow > frame_w or oy + oh > frame_h:
            continue
        cleaned_other_players.append((ox, oy, ow, oh))
    other_players = cleaned_other_players

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or np.isnan(fps):
        fps = 30.0

    # Use MJPEG in AVI (friendly for ffmpeg conversion)
    output_path = tempfile.mktemp(suffix=".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter with MJPG/AVI.")
    print(f"[VideoWriter] Using MJPG in AVI, fps: {fps}")

    # Initialize trackers
    player_tracker = PlayerTracker(collision_th)
    other_trackers = [PlayerTracker(collision_th) for _ in other_players]

    # First frame for init
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Could not read first frame for tracker initialization")

    player_tracker.init_tracking(frame, main_bbox)
    for tracker, bbox in zip(other_trackers, other_players):
        tracker.init_tracking(frame, bbox)

    player_positions = deque(maxlen=15)
    last_movement = None
    movement_counts = {"JUMPING": 0, "LANDING": 0, "TACKLE": 0, "COLLISION": 0}
    frame_counter = 0

    # Main loop
    with open(txt_output_path, "w") as txt_file:
        # write the first processed frame as well
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
                        if success and bbox is not None:
                            other_bboxes_frame.append(bbox)
                            ox, oy, ow, oh = [int(v) for v in bbox]
                            cv2.rectangle(
                                frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2
                            )
                            cv2.putText(
                                frame,
                                f"Player {idx + 1}",
                                (ox, oy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                # Main player
                tracking_result = player_tracker.update(frame)
                if tracking_result is None:
                    out.write(frame)
                    continue

                if len(tracking_result) == 3:
                    success, bbox, movement = tracking_result
                else:
                    success, bbox = tracking_result
                    movement = None

                if success and bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 0, 255), 2
                    )
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

                    center_x = (x + w / 2) / frame_width
                    center_y = (y + h / 2) / frame_height
                    norm_width = w / frame_width
                    norm_height = h / frame_height

                    # Movement with collisions
                    movement = player_tracker.movement_analyzer.update(
                        (x, y, w, h), other_bboxes_frame
                    )

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

                    risk_score = calculate_risk_score(movement_counts, frame_counter)
                    display_lines, total_line, risk_color = format_risk_display(
                        movement_counts, risk_score
                    )

                    txt_file.write(
                        f"{0 if movement is None else MOVEMENT_MAPPING[movement]} "
                        f"{center_x:.6f} {center_y:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}\n"
                    )

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

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path, txt_output_path
