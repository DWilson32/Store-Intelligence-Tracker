"""
detect.py — Main detection + tracking script for Store Intelligence Pipeline
Processes CCTV clips, detects people, tracks movement, and emits structured events.

Usage:
    python detect.py --clip <path_to_clip> --store_id STORE_BLR_002 \
                     --camera_id CAM_ENTRY_01 --layout store_layout.json \
                     --output events.jsonl
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Conditional imports with graceful degradation
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed — detection will use fallback mode")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pipeline.tracker import MultiCameraTracker
    from pipeline.emit import EventEmitter, SampleEventEmitter, EventType, age_to_bucket
except ImportError:
    from tracker import MultiCameraTracker
    from emit import EventEmitter, SampleEventEmitter, EventType, age_to_bucket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("detect")


# ---------------------------------------------------------------------------
# Staff Detection — uniform color heuristic + bounding-box region analysis
# ---------------------------------------------------------------------------

STAFF_UNIFORM_COLORS_HSV = [
    # (lower_bound, upper_bound) in HSV — tune per store
    (np.array([0, 0, 60]),   np.array([15, 40, 180])),   # beige / khaki
    (np.array([100, 60, 60]), np.array([130, 255, 255])), # blue uniform
    (np.array([0, 0, 200]),  np.array([180, 30, 255])),   # white uniform
    (np.array([0, 0, 0]),    np.array([180, 255, 55])),   # black uniform
    (np.array([140, 30, 80]), np.array([175, 255, 255])), # pink uniform (Store 2)
]


def detect_staff_by_uniform(frame: np.ndarray, bbox: tuple[int,int,int,int]) -> tuple[bool, float]:
    """
    Classify whether a detection is a staff member by analysing uniform colour
    in the torso region of the bounding box.

    Returns (is_staff: bool, confidence: float)
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    # Focus on torso region (mid 40% of bounding box height)
    torso_y1 = y1 + int(h * 0.30)
    torso_y2 = y1 + int(h * 0.70)

    torso_y1 = max(0, torso_y1)
    torso_y2 = min(frame.shape[0], torso_y2)
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)

    if torso_y2 <= torso_y1 or x2 <= x1:
        return False, 0.0

    torso = frame[torso_y1:torso_y2, x1:x2]
    if torso.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    total_pixels = torso.shape[0] * torso.shape[1]

    max_ratio = 0.0
    for lower, upper in STAFF_UNIFORM_COLORS_HSV:
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.sum(mask > 0) / total_pixels
        max_ratio = max(max_ratio, float(ratio))

    is_staff = max_ratio > 0.40
    confidence = min(max_ratio, 1.0)
    return is_staff, confidence


# ---------------------------------------------------------------------------
# Zone classifier — maps (x, y) centroid to zone_id using store_layout.json
# ---------------------------------------------------------------------------

class ZoneClassifier:
    def __init__(self, layout: dict, camera_id: str, store_id: Optional[str] = None):
        self.zones = []
        camera_data = get_camera_layout(layout, camera_id, store_id)
        for zone in camera_data.get("zones", []):
            self.zones.append({
                "zone_id": zone["zone_id"],
                "sku_zone": zone.get("sku_zone", zone["zone_id"]),
                "zone_name": zone.get("zone_name", zone["zone_id"]),
                "zone_type": zone.get("zone_type", "SHELF"),
                "is_revenue_zone": zone.get("is_revenue_zone", "Yes"),
                "polygon": np.array(zone["polygon"], dtype=np.float32),
            })
        logger.info(f"ZoneClassifier: loaded {len(self.zones)} zones for {camera_id}")

    def classify(self, cx: float, cy: float) -> Optional[dict]:
        """Returns zone info dict or None if point is not in any zone."""
        pt = (float(cx), float(cy))
        for z in self.zones:
            if cv2.pointPolygonTest(z["polygon"], pt, False) >= 0:
                return {
                    "zone_id": z["zone_id"],
                    "sku_zone": z["sku_zone"],
                    "zone_name": z["zone_name"],
                    "zone_type": z["zone_type"],
                    "is_revenue_zone": z["is_revenue_zone"],
                }
        return None


# ---------------------------------------------------------------------------
# Gender / Age estimation — lightweight heuristic based on body proportions
# ---------------------------------------------------------------------------

def estimate_gender_age(frame: np.ndarray, bbox: tuple[int,int,int,int]) -> tuple[Optional[str], Optional[int]]:
    """
    Lightweight gender/age estimation from person bounding box.
    Uses body proportion heuristics (height/width ratio, color histogram features).
    Not highly accurate — provides reasonable demographic signals.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None, None

    aspect = h / w
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(frame.shape[1], x2)
    y2c = min(frame.shape[0], y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))

    # Heuristic: higher color saturation correlates with female clothing
    # Higher aspect ratio (taller/narrower) skews male
    gender_score = 0.5
    if s_mean > 80:
        gender_score += 0.15
    if aspect < 2.8:
        gender_score += 0.10
    if h_mean > 10 and h_mean < 25:  # warm tones
        gender_score += 0.05

    gender = "F" if gender_score > 0.6 else "M"

    # Age estimation: use bbox area relative to frame as proxy
    bbox_area_ratio = (w * h) / (frame.shape[0] * frame.shape[1] + 1e-8)
    base_age = 28
    if bbox_area_ratio > 0.15:
        base_age = 32  # closer to camera = often adult
    elif bbox_area_ratio < 0.03:
        base_age = 22  # farther = often younger shopper

    # Add small variation based on appearance features
    age = base_age + int((h_mean % 7) - 3)
    age = max(18, min(65, age))

    return gender, age


# ---------------------------------------------------------------------------
# Face detection — check if face is visible in person bbox
# ---------------------------------------------------------------------------

_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def is_face_hidden(frame: np.ndarray, bbox: tuple[int,int,int,int]) -> bool:
    """Returns True if no face is detected within the person bounding box."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if y2 <= y1 or x2 <= x1:
        return True

    # Focus on upper 40% of bbox (head region)
    head_y2 = y1 + int((y2 - y1) * 0.4)
    head_crop = frame[y1:head_y2, x1:x2]
    if head_crop.size == 0:
        return True

    gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    return len(faces) == 0


# ---------------------------------------------------------------------------
# Group detection — spatial proximity + temporal window
# ---------------------------------------------------------------------------

class GroupDetector:
    """Detects groups of people entering together based on spatial proximity and timing."""
    PROXIMITY_THRESHOLD = 150  # pixels
    TIME_WINDOW_FRAMES = 45    # ~3 seconds at 15fps

    def __init__(self):
        self._recent_entries: list[dict] = []  # {visitor_id, cx, cy, frame_idx, group_id}
        self._group_counter = 0
        self._visitor_groups: dict[str, tuple[str, int]] = {}  # visitor_id -> (group_id, group_size)

    def check_group(self, visitor_id: str, cx: float, cy: float, frame_idx: int) -> tuple[Optional[str], Optional[int]]:
        """Check if this visitor is part of a group. Returns (group_id, group_size) or (None, None)."""
        # Clean old entries
        self._recent_entries = [
            e for e in self._recent_entries
            if frame_idx - e["frame_idx"] <= self.TIME_WINDOW_FRAMES
        ]

        # Check proximity against recent entries
        matching_entry = None
        for entry in self._recent_entries:
            dist = ((cx - entry["cx"]) ** 2 + (cy - entry["cy"]) ** 2) ** 0.5
            if dist < self.PROXIMITY_THRESHOLD:
                matching_entry = entry
                break

        if matching_entry is None:
            # No group match — create new potential group marker
            self._recent_entries.append({
                "visitor_id": visitor_id, "cx": cx, "cy": cy,
                "frame_idx": frame_idx, "group_id": None,
            })
            return None, None

        matched_group = matching_entry.get("group_id")
        if matched_group is None:
            # First pairing — assign new group_id to both
            self._group_counter += 1
            matched_group = f"G_{self._group_counter}"
            for entry in self._recent_entries:
                dist = ((cx - entry["cx"]) ** 2 + (cy - entry["cy"]) ** 2) ** 0.5
                if dist < self.PROXIMITY_THRESHOLD:
                    entry["group_id"] = matched_group
                    self._visitor_groups[entry["visitor_id"]] = (matched_group, 2)

        # Add this visitor to the group
        self._recent_entries.append({
            "visitor_id": visitor_id, "cx": cx, "cy": cy,
            "frame_idx": frame_idx, "group_id": matched_group,
        })

        # Count group size
        group_members = set(
            e["visitor_id"] for e in self._recent_entries
            if e.get("group_id") == matched_group
        )
        group_size = len(group_members)
        self._visitor_groups[visitor_id] = (matched_group, group_size)

        # Update all members with correct size
        for vid in group_members:
            self._visitor_groups[vid] = (matched_group, group_size)

        return matched_group, group_size

    def get_group(self, visitor_id: str) -> tuple[Optional[str], Optional[int]]:
        return self._visitor_groups.get(visitor_id, (None, None))


# ---------------------------------------------------------------------------
# Entry / Exit direction classifier
# ---------------------------------------------------------------------------

class EntryExitClassifier:
    """
    Determines ENTRY vs EXIT by comparing the centroid trajectory across
    the entry threshold line defined in store_layout.json.

    Threshold is a horizontal or vertical line segment. Movement crossing
    from outside-to-inside = ENTRY, inside-to-outside = EXIT.
    """
    def __init__(self, layout: dict, camera_id: str, store_id: Optional[str] = None):
        cam = get_camera_layout(layout, camera_id, store_id)
        self.threshold = cam.get("entry_threshold", None)
        # threshold format: {"axis": "y", "value": 400, "inside": "below"}
        # or              : {"axis": "x", "value": 200, "inside": "right"}

    def classify_crossing(self, prev_cy: float, curr_cy: float,
                           prev_cx: float, curr_cx: float) -> Optional[str]:
        if self.threshold is None:
            return None
        axis = self.threshold.get("axis", "y")
        value = self.threshold.get("value", 0)
        inside = self.threshold.get("inside", "below")

        if axis == "y":
            if inside == "below":
                if prev_cy <= value < curr_cy:
                    return EventType.ENTRY
                elif prev_cy >= value > curr_cy:
                    return EventType.EXIT
            else:  # above
                if prev_cy >= value > curr_cy:
                    return EventType.ENTRY
                elif prev_cy <= value < curr_cy:
                    return EventType.EXIT
        else:  # x axis
            if inside == "right":
                if prev_cx <= value < curr_cx:
                    return EventType.ENTRY
                elif prev_cx >= value > curr_cx:
                    return EventType.EXIT
            else:
                if prev_cx >= value > curr_cx:
                    return EventType.ENTRY
                elif prev_cx <= value < curr_cx:
                    return EventType.EXIT
        return None


def get_camera_layout(layout: dict, camera_id: str, store_id: Optional[str] = None) -> dict:
    """Return a camera definition from flat or nested store layout JSON."""
    if store_id and "stores" in layout:
        store_data = layout["stores"].get(store_id)
        if store_data and camera_id in store_data.get("cameras", {}):
            return store_data["cameras"][camera_id]

    if camera_id in layout.get("cameras", {}):
        return layout["cameras"][camera_id]

    for store in layout.get("stores", {}).values():
        cameras = store.get("cameras", {})
        if camera_id in cameras:
            return cameras[camera_id]

    return {}


# ---------------------------------------------------------------------------
# Per-track session state
# ---------------------------------------------------------------------------

class TrackSession:
    def __init__(self, visitor_id: str):
        self.visitor_id = visitor_id
        self.entered = False
        self.exited = False
        self.current_zone: Optional[str] = None
        self.current_zone_info: Optional[dict] = None  # full zone metadata
        self.zone_entry_frame: Optional[int] = None
        self.last_dwell_emit_frame: Optional[int] = None
        self.session_seq = 0
        self.in_billing = False
        self.billing_entry_frame: Optional[int] = None
        self.billing_join_ts: Optional[str] = None  # queue join timestamp
        self.billing_zone_id: Optional[str] = None
        self.billing_zone_name: Optional[str] = None
        self.prev_cx: Optional[float] = None
        self.prev_cy: Optional[float] = None
        self.last_cx: float = 0.0  # most recent centroid
        self.last_cy: float = 0.0
        self.frames_seen = 0
        # Demographics (cached per session)
        self.gender: Optional[str] = None
        self.age: Optional[int] = None
        self.face_hidden: bool = False
        self.raw_track_id: int = 0  # numeric track_id for sample format

    def next_seq(self) -> int:
        self.session_seq += 1
        return self.session_seq


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class StoreDetector:
    DWELL_EMIT_INTERVAL_SECONDS = 30
    PERSON_CLASS_ID = 0  # COCO class 0 = person
    CONFIDENCE_THRESHOLD = float(os.environ.get("DETECTION_CONFIDENCE_MIN", "0.05"))
    MIN_BBOX_AREA = 1500  # pixels² — ignore tiny false positives

    def __init__(
        self,
        clip_path: str,
        store_id: str,
        camera_id: str,
        layout_path: str,
        output_path: str,
        clip_start_time: Optional[str] = None,
        fps_override: Optional[float] = None,
        tracker: Optional[MultiCameraTracker] = None,
        output_format: str = "internal",  # "internal" or "sample"
    ):
        self.clip_path = clip_path
        self.store_id = store_id
        self.camera_id = camera_id
        self.output_path = output_path
        self.output_format = output_format

        with open(layout_path) as f:
            self.layout = json.load(f)

        self.clip_start_dt = (
            datetime.fromisoformat(clip_start_time.replace("Z", "+00:00"))
            if clip_start_time
            else datetime.now(timezone.utc)
        )

        self.zone_classifier = ZoneClassifier(self.layout, camera_id, store_id)
        self.entry_exit_clf = EntryExitClassifier(self.layout, camera_id, store_id)
        self.tracker = tracker or MultiCameraTracker(store_id=store_id)
        self.group_detector = GroupDetector()

        # Dual emitter: internal format for ST1008, sample format for evaluation
        if output_format == "sample":
            self.sample_emitter = SampleEventEmitter(
                store_id=store_id, camera_id=camera_id, output_path=output_path
            )
            self.emitter = None  # not used in sample mode
        else:
            self.emitter = EventEmitter(
                store_id=store_id, camera_id=camera_id, output_path=output_path
            )
            self.sample_emitter = None

        if YOLO_AVAILABLE:
            model_path = os.environ.get("YOLO_MODEL", "yolov8n.pt")
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
        else:
            self.model = None
            logger.warning("Running in MOCK detection mode — no YOLO available")

        self.fps_override = fps_override
        self.sessions: dict[str, TrackSession] = {}  # track_id → TrackSession

    def _frame_to_timestamp(self, frame_idx: int, fps: float) -> str:
        offset_s = frame_idx / fps
        ts = self.clip_start_dt + timedelta(seconds=offset_s)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _detect_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO on one frame. Returns list of {bbox, conf, cls}.
        Falls back to empty list if model unavailable.
        """
        if self.model is None:
            return []

        results = self.model(frame, classes=[self.PERSON_CLASS_ID], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area < self.MIN_BBOX_AREA:
                    continue
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": int(box.cls[0]),
                })
        return detections

    def _get_or_create_session(self, visitor_id: str) -> TrackSession:
        if visitor_id not in self.sessions:
            self.sessions[visitor_id] = TrackSession(visitor_id)
        return self.sessions[visitor_id]

    def _handle_zone_transition(self, session: TrackSession, new_zone_info: Optional[dict],
                                 frame_idx: int, timestamp: str, fps: float,
                                 bbox: tuple, conf: float, is_staff: bool):
        """Emit ZONE_ENTER, ZONE_EXIT, ZONE_DWELL events on zone transitions."""
        new_zone_id = new_zone_info["zone_id"] if new_zone_info else None
        sku_zone = new_zone_info["sku_zone"] if new_zone_info else None
        zone_name = new_zone_info.get("zone_name", new_zone_id) if new_zone_info else None
        zone_type = new_zone_info.get("zone_type", "SHELF") if new_zone_info else None
        is_rev = new_zone_info.get("is_revenue_zone", "Yes") if new_zone_info else "Yes"

        if new_zone_id != session.current_zone:
            # Zone exit
            if session.current_zone is not None:
                dwell_ms = int(
                    (frame_idx - (session.zone_entry_frame or frame_idx)) / fps * 1000
                )
                prev_info = session.current_zone_info or {}
                if self.output_format == "sample" and self.sample_emitter:
                    self.sample_emitter.emit_zone(
                        event_type=EventType.ZONE_EXIT,
                        track_id=session.raw_track_id,
                        timestamp=timestamp,
                        zone_id=session.current_zone,
                        zone_name=prev_info.get("zone_name", session.current_zone),
                        zone_type=prev_info.get("zone_type", "SHELF"),
                        is_revenue_zone=prev_info.get("is_revenue_zone", "Yes"),
                        cx=session.last_cx, cy=session.last_cy,
                        gender=session.gender, age=session.age,
                    )
                elif self.emitter:
                    self.emitter.emit(
                        event_type=EventType.ZONE_EXIT,
                        visitor_id=session.visitor_id,
                        timestamp=timestamp,
                        zone_id=session.current_zone,
                        dwell_ms=dwell_ms,
                        is_staff=is_staff,
                        confidence=conf,
                        session_seq=session.next_seq(),
                        metadata={"sku_zone": sku_zone, "session_seq": session.session_seq},
                    )

            # Zone enter
            if new_zone_id is not None:
                if self.output_format == "sample" and self.sample_emitter:
                    self.sample_emitter.emit_zone(
                        event_type=EventType.ZONE_ENTER,
                        track_id=session.raw_track_id,
                        timestamp=timestamp,
                        zone_id=new_zone_id,
                        zone_name=zone_name or new_zone_id,
                        zone_type=zone_type or "SHELF",
                        is_revenue_zone=is_rev,
                        cx=session.last_cx, cy=session.last_cy,
                        gender=session.gender, age=session.age,
                    )
                elif self.emitter:
                    self.emitter.emit(
                        event_type=EventType.ZONE_ENTER,
                        visitor_id=session.visitor_id,
                        timestamp=timestamp,
                        zone_id=new_zone_id,
                        dwell_ms=0,
                        is_staff=is_staff,
                        confidence=conf,
                        session_seq=session.next_seq(),
                        metadata={"sku_zone": sku_zone, "session_seq": session.session_seq},
                    )
                session.zone_entry_frame = frame_idx
                session.last_dwell_emit_frame = frame_idx

            session.current_zone = new_zone_id
            session.current_zone_info = new_zone_info

        else:
            # Still in same zone — check if we should emit ZONE_DWELL
            if new_zone_id is not None and session.last_dwell_emit_frame is not None:
                frames_since_dwell = frame_idx - session.last_dwell_emit_frame
                if frames_since_dwell >= int(fps * self.DWELL_EMIT_INTERVAL_SECONDS):
                    dwell_ms = int(frames_since_dwell / fps * 1000)
                    if self.emitter:
                        self.emitter.emit(
                            event_type=EventType.ZONE_DWELL,
                            visitor_id=session.visitor_id,
                            timestamp=timestamp,
                            zone_id=new_zone_id,
                            dwell_ms=dwell_ms,
                            is_staff=is_staff,
                            confidence=conf,
                            session_seq=session.next_seq(),
                            metadata={"sku_zone": sku_zone, "session_seq": session.session_seq},
                        )
                    session.last_dwell_emit_frame = frame_idx

    def process(self):
        cap = cv2.VideoCapture(self.clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.clip_path}")

        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {self.clip_path} | FPS={fps:.1f} | Frames={total_frames}")

        frame_idx = 0
        # Process every other frame for performance (still sub-second resolution)
        FRAME_SKIP = 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            timestamp = self._frame_to_timestamp(frame_idx, fps)
            detections = self._detect_frame(frame)

            # Pass detections to tracker — get back list of {track_id, bbox, conf}
            tracked = self.tracker.update(
                detections=detections,
                frame=frame,
                frame_idx=frame_idx,
                camera_id=self.camera_id,
            )

            for track in tracked:
                track_id = track["track_id"]
                bbox = track["bbox"]
                conf = track["conf"]

                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                visitor_id = self.tracker.get_visitor_id(
                    track_id,
                    embedding=track.get("embedding"),
                    frame=frame,
                    bbox=bbox,
                )
                session = self._get_or_create_session(visitor_id)
                session.last_cx = cx
                session.last_cy = cy

                # Extract raw numeric track_id for sample format
                raw_tid = track.get("raw_track_id", 0)
                session.raw_track_id = raw_tid

                # Demographics estimation (cache per session, update periodically)
                if session.gender is None or session.frames_seen % 30 == 0:
                    gender, age = estimate_gender_age(frame, bbox)
                    if gender:
                        session.gender = gender
                    if age:
                        session.age = age
                    session.face_hidden = is_face_hidden(frame, bbox)

                is_staff, staff_conf = detect_staff_by_uniform(frame, bbox)
                effective_conf = conf * (1 - 0.3 * int(is_staff))  # slightly lower conf for staff classifications

                # Entry / Exit detection
                if session.prev_cy is not None:
                    crossing = self.entry_exit_clf.classify_crossing(
                        session.prev_cy, cy, session.prev_cx or cx, cx
                    )
                    if crossing == EventType.ENTRY and not session.entered:
                        session.entered = True
                        event_type = EventType.ENTRY
                        # Check re-entry
                        if self.tracker.is_reentry(visitor_id):
                            event_type = EventType.REENTRY

                        # Group detection
                        group_id, group_size = self.group_detector.check_group(
                            visitor_id, cx, cy, frame_idx
                        )

                        if self.output_format == "sample" and self.sample_emitter:
                            self.sample_emitter.emit_entry_exit(
                                event_type=event_type,
                                visitor_id=visitor_id,
                                timestamp=timestamp,
                                is_staff=is_staff,
                                gender_pred=session.gender,
                                age_pred=session.age,
                                is_face_hidden=session.face_hidden,
                                group_id=group_id,
                                group_size=group_size,
                            )
                        elif self.emitter:
                            self.emitter.emit(
                                event_type=event_type,
                                visitor_id=visitor_id,
                                timestamp=timestamp,
                                zone_id=None,
                                dwell_ms=0,
                                is_staff=is_staff,
                                confidence=effective_conf,
                                session_seq=session.next_seq(),
                                metadata={"session_seq": session.session_seq},
                            )

                    elif crossing == EventType.EXIT and not session.exited:
                        session.exited = True

                        if self.output_format == "sample" and self.sample_emitter:
                            group_id, group_size = self.group_detector.get_group(visitor_id)
                            self.sample_emitter.emit_entry_exit(
                                event_type=EventType.EXIT,
                                visitor_id=visitor_id,
                                timestamp=timestamp,
                                is_staff=is_staff,
                                gender_pred=session.gender,
                                age_pred=session.age,
                                is_face_hidden=session.face_hidden,
                                group_id=group_id,
                                group_size=group_size,
                            )
                        elif self.emitter:
                            self.emitter.emit(
                                event_type=EventType.EXIT,
                                visitor_id=visitor_id,
                                timestamp=timestamp,
                                zone_id=None,
                                dwell_ms=0,
                                is_staff=is_staff,
                                confidence=effective_conf,
                                session_seq=session.next_seq(),
                                metadata={"session_seq": session.session_seq},
                            )
                        self.tracker.mark_exited(visitor_id)

                session.prev_cx = cx
                session.prev_cy = cy
                session.frames_seen += 1

                # Zone classification
                zone_info = self.zone_classifier.classify(cx, cy)
                self._handle_zone_transition(
                    session=session,
                    new_zone_info=zone_info,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    fps=fps,
                    bbox=bbox,
                    conf=effective_conf,
                    is_staff=is_staff,
                )

                # Billing queue logic
                if zone_info and "BILLING" in zone_info.get("zone_id", "").upper():
                    self.tracker.update_billing_presence(self.camera_id, track_id, True)
                    queue_depth = self.tracker.get_billing_queue_depth(self.camera_id)
                    if queue_depth > 0 and not session.in_billing:
                        session.in_billing = True
                        session.billing_entry_frame = frame_idx
                        session.billing_join_ts = timestamp
                        session.billing_zone_id = zone_info["zone_id"]
                        session.billing_zone_name = zone_info.get("zone_name", zone_info["zone_id"])
                        session.queue_position_at_join = queue_depth

                        if self.output_format != "sample" and self.emitter:
                            self.emitter.emit(
                                event_type=EventType.BILLING_QUEUE_JOIN,
                                visitor_id=visitor_id,
                                timestamp=timestamp,
                                zone_id=zone_info["zone_id"],
                                dwell_ms=0,
                                is_staff=is_staff,
                                confidence=effective_conf,
                                session_seq=session.next_seq(),
                                metadata={
                                    "queue_depth": queue_depth,
                                    "sku_zone": zone_info.get("sku_zone"),
                                    "session_seq": session.session_seq,
                                },
                            )
                elif session.in_billing:
                    # Left billing zone — emit queue event
                    session.in_billing = False
                    self.tracker.update_billing_presence(self.camera_id, track_id, False)

                    wait_seconds = int(
                        (frame_idx - (session.billing_entry_frame or frame_idx)) / fps
                    )
                    queue_pos = getattr(session, "queue_position_at_join", 1)

                    if self.output_format == "sample" and self.sample_emitter:
                        # In sample format, we emit queue_completed/queue_abandoned
                        # Determine if abandoned: if they left quickly and no POS correlation
                        # For now, emit as queue_completed (POS correlation resolves later)
                        abandoned = wait_seconds < 5  # heuristic: very short = abandoned
                        self.sample_emitter.emit_queue(
                            event_type="queue_abandoned" if abandoned else "queue_completed",
                            track_id=session.raw_track_id,
                            zone_id=session.billing_zone_id or "BILLING",
                            zone_name=session.billing_zone_name or "Billing Counter Queue",
                            queue_join_ts=session.billing_join_ts or timestamp,
                            queue_served_ts=None if abandoned else timestamp,
                            queue_exit_ts=timestamp,
                            wait_seconds=wait_seconds,
                            queue_position_at_join=queue_pos,
                            abandoned=abandoned,
                            cx=cx, cy=cy,
                            gender=session.gender, age=session.age,
                        )
                    elif self.emitter:
                        self.emitter.emit(
                            event_type=EventType.BILLING_QUEUE_ABANDON,
                            visitor_id=visitor_id,
                            timestamp=timestamp,
                            zone_id=None,
                            dwell_ms=0,
                            is_staff=is_staff,
                            confidence=effective_conf,
                            session_seq=session.next_seq(),
                            metadata={"session_seq": session.session_seq},
                        )

            if frame_idx % 300 == 0:
                pct = frame_idx / max(total_frames, 1) * 100
                logger.info(f"Progress: {frame_idx}/{total_frames} frames ({pct:.1f}%)")

        cap.release()
        active_emitter = self.sample_emitter if self.output_format == "sample" else self.emitter
        if active_emitter:
            active_emitter.flush()
            total_events = active_emitter.event_count
        else:
            total_events = 0
        logger.info(f"Done. Emitted {total_events} events → {self.output_path}")
        return total_events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--clip", required=True, help="Path to CCTV clip")
    parser.add_argument("--store_id", required=True, help="Store ID (e.g. STORE_BLR_002)")
    parser.add_argument("--camera_id", required=True, help="Camera ID (e.g. CAM_ENTRY_01)")
    parser.add_argument("--layout", required=True, help="Path to store_layout.json")
    parser.add_argument("--output", required=True, help="Output .jsonl file for events")
    parser.add_argument("--start_time", default=None,
                        help="ISO-8601 UTC timestamp of clip start (e.g. 2026-03-03T14:00:00Z)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS detection")
    parser.add_argument("--output_format", default="internal", choices=["internal", "sample"],
                        help="Output event format: 'internal' (our schema) or 'sample' (evaluator schema)")
    args = parser.parse_args()

    detector = StoreDetector(
        clip_path=args.clip,
        store_id=args.store_id,
        camera_id=args.camera_id,
        layout_path=args.layout,
        output_path=args.output,
        clip_start_time=args.start_time,
        fps_override=args.fps,
        output_format=args.output_format,
    )
    n = detector.process()
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
