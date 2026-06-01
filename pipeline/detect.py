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
import sys
import time
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
    from pipeline.emit import EventEmitter, EventType
except ImportError:
    from tracker import MultiCameraTracker
    from emit import EventEmitter, EventType

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

    is_staff = max_ratio > 0.55
    confidence = min(max_ratio, 1.0)
    return is_staff, confidence


# ---------------------------------------------------------------------------
# Zone classifier — maps (x, y) centroid to zone_id using store_layout.json
# ---------------------------------------------------------------------------

class ZoneClassifier:
    def __init__(self, layout: dict, camera_id: str):
        self.zones = []
        camera_data = get_camera_layout(layout, camera_id)
        for zone in camera_data.get("zones", []):
            self.zones.append({
                "zone_id": zone["zone_id"],
                "sku_zone": zone.get("sku_zone", zone["zone_id"]),
                "polygon": np.array(zone["polygon"], dtype=np.float32),
            })
        logger.info(f"ZoneClassifier: loaded {len(self.zones)} zones for {camera_id}")

    def classify(self, cx: float, cy: float) -> Optional[dict]:
        """Returns zone info dict or None if point is not in any zone."""
        pt = (float(cx), float(cy))
        for z in self.zones:
            if cv2.pointPolygonTest(z["polygon"], pt, False) >= 0:
                return {"zone_id": z["zone_id"], "sku_zone": z["sku_zone"]}
        return None


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
    def __init__(self, layout: dict, camera_id: str):
        cam = get_camera_layout(layout, camera_id)
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


def get_camera_layout(layout: dict, camera_id: str) -> dict:
    """Return a camera definition from flat or nested store layout JSON."""
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
        self.zone_entry_frame: Optional[int] = None
        self.last_dwell_emit_frame: Optional[int] = None
        self.session_seq = 0
        self.in_billing = False
        self.billing_entry_frame: Optional[int] = None
        self.prev_cx: Optional[float] = None
        self.prev_cy: Optional[float] = None
        self.frames_seen = 0

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
    ):
        self.clip_path = clip_path
        self.store_id = store_id
        self.camera_id = camera_id
        self.output_path = output_path

        with open(layout_path) as f:
            self.layout = json.load(f)

        self.clip_start_dt = (
            datetime.fromisoformat(clip_start_time.replace("Z", "+00:00"))
            if clip_start_time
            else datetime.now(timezone.utc)
        )

        self.zone_classifier = ZoneClassifier(self.layout, camera_id)
        self.entry_exit_clf = EntryExitClassifier(self.layout, camera_id)
        self.tracker = tracker or MultiCameraTracker(store_id=store_id)
        self.emitter = EventEmitter(store_id=store_id, camera_id=camera_id, output_path=output_path)

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

        if new_zone_id != session.current_zone:
            # Zone exit
            if session.current_zone is not None:
                dwell_ms = int(
                    (frame_idx - (session.zone_entry_frame or frame_idx)) / fps * 1000
                )
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

        else:
            # Still in same zone — check if we should emit ZONE_DWELL
            if new_zone_id is not None and session.last_dwell_emit_frame is not None:
                frames_since_dwell = frame_idx - session.last_dwell_emit_frame
                if frames_since_dwell >= int(fps * self.DWELL_EMIT_INTERVAL_SECONDS):
                    dwell_ms = int(frames_since_dwell / fps * 1000)
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
                    # Left billing zone — check if purchase followed (done in API via POS correlation)
                    # Emit abandon candidate; API will resolve post-POS correlation
                    session.in_billing = False
                    self.tracker.update_billing_presence(self.camera_id, track_id, False)
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
        self.emitter.flush()
        total_events = self.emitter.event_count
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
    args = parser.parse_args()

    detector = StoreDetector(
        clip_path=args.clip,
        store_id=args.store_id,
        camera_id=args.camera_id,
        layout_path=args.layout,
        output_path=args.output,
        clip_start_time=args.start_time,
        fps_override=args.fps,
    )
    n = detector.process()
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
