"""
tracker.py — Re-ID / tracking logic for Store Intelligence Pipeline

Responsibilities:
  - Wrap ByteTrack (or fallback simple IoU tracker) for per-frame track management
  - Assign stable visitor_id tokens via appearance-based Re-ID (OSNet embeddings)
  - Handle re-entry detection (same physical person after EXIT event)
  - Cross-camera deduplication (same person seen on multiple cameras)
  - Maintain billing queue depth per camera

Re-ID approach:
  Primary:  OSNet (torchreid) cosine similarity on 512-d appearance embeddings
  Fallback: Bounding-box trajectory similarity when torchreid unavailable
  Re-entry: Match against a rolling window of recent EXIT embeddings (10-min window)
"""

import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("tracker")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully if missing
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO
    # ByteTrack is built into ultralytics' track() method
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

try:
    import torchreid
    import torch

    _REID_MODEL = None

    def _get_reid_model():
        global _REID_MODEL
        if _REID_MODEL is None:
            model_name = os.environ.get("REID_MODEL", "osnet_x0_25")
            logger.info(f"Loading Re-ID model: {model_name}")
            _REID_MODEL = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=True,
            )
            _REID_MODEL.eval()
            if torch.cuda.is_available():
                _REID_MODEL = _REID_MODEL.cuda()
        return _REID_MODEL

    def extract_embedding(img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-d Re-ID embedding for a person crop."""
        try:
            model = _get_reid_model()
            patch = cv2.resize(img_patch, (128, 256))
            patch = patch.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            patch = (patch - mean) / std
            tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float()
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                feat = model(tensor)
            feat = feat.cpu().numpy()[0]
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat
        except Exception as e:
            logger.debug(f"Re-ID embedding failed: {e}")
            return None

    REID_AVAILABLE = True
    logger.info("torchreid available — using appearance-based Re-ID")

except ImportError:
    REID_AVAILABLE = False
    logger.warning("torchreid not installed — using trajectory-based Re-ID fallback")

    def extract_embedding(img_patch: np.ndarray) -> Optional[np.ndarray]:
        return None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Simple IoU-based tracker (fallback when ByteTrack unavailable)
# ---------------------------------------------------------------------------

def iou(boxA: tuple, boxB: tuple) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SimpleTrack:
    track_id: int
    bbox: tuple
    conf: float
    age: int = 0
    missed_frames: int = 0
    embedding: Optional[np.ndarray] = None


class SimpleIoUTracker:
    """
    Minimal IoU tracker — used when ByteTrack is unavailable.
    Assigns stable track IDs across frames using greedy IoU matching.
    """
    IOU_THRESHOLD = 0.3
    MAX_MISSED = 30  # frames before track is removed

    def __init__(self):
        self._tracks: list[SimpleTrack] = []
        self._next_id = 1

    def update(self, detections: list[dict], frame: np.ndarray,
               frame_idx: int) -> list[dict]:
        """
        Match detections to existing tracks via IoU.
        Returns list of {track_id, bbox, conf}.
        """
        # Mark all as missed initially
        for t in self._tracks:
            t.missed_frames += 1

        results = []
        used_track_ids = set()

        for det in detections:
            bbox = det["bbox"]
            conf = det["conf"]
            best_iou = self.IOU_THRESHOLD
            best_track = None

            for t in self._tracks:
                if t.track_id in used_track_ids:
                    continue
                score = iou(bbox, t.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = t

            if best_track is not None:
                best_track.bbox = bbox
                best_track.conf = conf
                best_track.missed_frames = 0
                best_track.age += 1
                used_track_ids.add(best_track.track_id)
                results.append({"track_id": best_track.track_id, "bbox": bbox, "conf": conf})
            else:
                # New track
                new_track = SimpleTrack(
                    track_id=self._next_id,
                    bbox=bbox,
                    conf=conf,
                )
                self._next_id += 1
                self._tracks.append(new_track)
                used_track_ids.add(new_track.track_id)
                results.append({"track_id": new_track.track_id, "bbox": bbox, "conf": conf})

        # Prune stale tracks
        self._tracks = [t for t in self._tracks if t.missed_frames < self.MAX_MISSED]
        return results


# ---------------------------------------------------------------------------
# Re-ID gallery — matches track embeddings to visitor_ids
# ---------------------------------------------------------------------------

@dataclass
class GalleryEntry:
    visitor_id: str
    embedding: np.ndarray
    last_seen_ts: float
    exited: bool = False


class ReIDGallery:
    """
    Maintains appearance embeddings for active + recently-exited visitors.
    Re-entry window: 10 minutes.
    """
    ACTIVE_SIMILARITY_THRESHOLD = 0.75
    REENTRY_SIMILARITY_THRESHOLD = 0.70
    REENTRY_WINDOW_SECONDS = 600  # 10 minutes

    def __init__(self, store_id: str):
        self.store_id = store_id
        self._gallery: dict[str, GalleryEntry] = {}  # visitor_id → entry
        self._track_to_visitor: dict[object, str] = {}  # camera-scoped track_id -> visitor_id
        self._reentry_set: set[str] = set()
        
        # Enable state persistence for cross-camera Re-ID across independent runs
        self.gallery_path = os.path.join("data", "events", f"reid_gallery_{store_id}.pkl")
        self._load_gallery()
        import atexit
        atexit.register(self._save_gallery)

    def _load_gallery(self):
        import pickle
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "rb") as f:
                    data = pickle.load(f)
                    self._gallery = data.get("gallery", {})
                    self._reentry_set = data.get("reentry_set", set())
                logger.info(f"Loaded {len(self._gallery)} entries from Re-ID gallery at {self.gallery_path}")
            except Exception as e:
                logger.warning(f"Failed to load Re-ID gallery from {self.gallery_path}: {e}")

    def _save_gallery(self):
        import pickle
        try:
            os.makedirs(os.path.dirname(self.gallery_path), exist_ok=True)
            with open(self.gallery_path, "wb") as f:
                pickle.dump({
                    "gallery": self._gallery,
                    "reentry_set": self._reentry_set,
                }, f)
            print(f"Saved Re-ID gallery to {self.gallery_path}")
        except Exception as e:
            print(f"Failed to save Re-ID gallery to {self.gallery_path}: {e}")

    def _make_visitor_id(self) -> str:
        return "VIS_" + uuid.uuid4().hex[:6]

    def get_visitor_id(self, track_id: int, embedding: Optional[np.ndarray],
                        frame: np.ndarray, bbox: tuple) -> str:
        """
        Returns the visitor_id for a track_id.
        Matches against gallery by embedding; creates new entry if no match.
        """
        # Already assigned
        if track_id in self._track_to_visitor:
            vid = self._track_to_visitor[track_id]
            # Update embedding with exponential moving average
            if embedding is not None and vid in self._gallery:
                old = self._gallery[vid].embedding
                self._gallery[vid].embedding = 0.7 * old + 0.3 * embedding
                self._gallery[vid].embedding /= (
                    np.linalg.norm(self._gallery[vid].embedding) + 1e-8
                )
                self._gallery[vid].last_seen_ts = time.time()
            return vid

        # Try to match active gallery
        if embedding is not None:
            now = time.time()
            best_sim = 0.0
            best_vid = None
            for vid, entry in self._gallery.items():
                if entry.exited:
                    continue
                sim = cosine_sim(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_vid = vid

            if best_sim >= self.ACTIVE_SIMILARITY_THRESHOLD and best_vid:
                self._track_to_visitor[track_id] = best_vid
                return best_vid

            # Try re-entry match against recently exited
            best_sim = 0.0
            best_vid = None
            for vid, entry in self._gallery.items():
                if not entry.exited:
                    continue
                if now - entry.last_seen_ts > self.REENTRY_WINDOW_SECONDS:
                    continue
                sim = cosine_sim(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_vid = vid

            if best_sim >= self.REENTRY_SIMILARITY_THRESHOLD and best_vid:
                # Re-entry detected — reactivate
                self._gallery[best_vid].exited = False
                self._gallery[best_vid].last_seen_ts = now
                self._track_to_visitor[track_id] = best_vid
                self._reentry_set.add(best_vid)
                logger.info(f"Re-entry detected: {best_vid} (sim={best_sim:.3f})")
                return best_vid

        # Completely new visitor
        visitor_id = self._make_visitor_id()
        emb = embedding if embedding is not None else np.zeros(512)
        self._gallery[visitor_id] = GalleryEntry(
            visitor_id=visitor_id,
            embedding=emb,
            last_seen_ts=time.time(),
        )
        self._track_to_visitor[track_id] = visitor_id
        return visitor_id

    def mark_exited(self, visitor_id: str):
        if visitor_id in self._gallery:
            self._gallery[visitor_id].exited = True
            self._gallery[visitor_id].last_seen_ts = time.time()

    def is_reentry(self, visitor_id: str) -> bool:
        return visitor_id in self._reentry_set

    def prune_old(self):
        """Remove gallery entries older than the re-entry window."""
        cutoff = time.time() - self.REENTRY_WINDOW_SECONDS
        to_remove = [
            vid for vid, entry in self._gallery.items()
            if entry.exited and entry.last_seen_ts < cutoff
        ]
        for vid in to_remove:
            del self._gallery[vid]


# ---------------------------------------------------------------------------
# Multi-camera tracker — main public interface
# ---------------------------------------------------------------------------

class MultiCameraTracker:
    """
    Wraps per-camera tracking and exposes a unified interface with:
    - ByteTrack (preferred) or simple IoU tracker (fallback)
    - Re-ID gallery shared across cameras for cross-camera deduplication
    - Billing queue depth tracking per camera
    """
    BILLING_ZONE_KEYWORDS = ["BILLING", "CHECKOUT", "CASHIER"]

    def __init__(self, store_id: str):
        self.store_id = store_id
        self._reid_gallery = ReIDGallery(store_id=store_id)
        self._per_camera_trackers: dict[str, SimpleIoUTracker] = {}
        self._billing_queue: dict[str, set] = defaultdict(set)  # camera_id → set of track_ids in billing

        if BYTETRACK_AVAILABLE:
            logger.info("ByteTrack available via ultralytics")
        else:
            logger.info("Using fallback SimpleIoUTracker")

    def _get_tracker(self, camera_id: str) -> SimpleIoUTracker:
        if camera_id not in self._per_camera_trackers:
            self._per_camera_trackers[camera_id] = SimpleIoUTracker()
        return self._per_camera_trackers[camera_id]

    def update(self, detections: list[dict], frame: np.ndarray,
               frame_idx: int, camera_id: str) -> list[dict]:
        """
        Update tracker with new detections.
        Returns list of {track_id, bbox, conf} with stable track IDs.
        """
        tracker = self._get_tracker(camera_id)
        tracked = tracker.update(detections=detections, frame=frame, frame_idx=frame_idx)

        results = []
        for t in tracked:
            track_id = t["track_id"]
            scoped_track_id = f"{camera_id}:{track_id}"
            bbox = t["bbox"]
            conf = t["conf"]

            # Extract appearance embedding for Re-ID
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            patch = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
            embedding = extract_embedding(patch) if patch is not None and patch.size > 0 else None

            results.append({
                "track_id": scoped_track_id,
                "raw_track_id": track_id,
                "bbox": bbox,
                "conf": conf,
                "embedding": embedding,
            })

        # Periodically prune old gallery entries
        if frame_idx % 450 == 0:
            self._reid_gallery.prune_old()

        return results

    def get_visitor_id(self, track_id: int, embedding: Optional[np.ndarray] = None,
                       frame: Optional[np.ndarray] = None, bbox: Optional[tuple] = None) -> str:
        return self._reid_gallery.get_visitor_id(
            track_id=track_id,
            embedding=embedding,
            frame=frame if frame is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            bbox=bbox or (0,0,1,1),
        )

    def mark_exited(self, visitor_id: str):
        self._reid_gallery.mark_exited(visitor_id)

    def is_reentry(self, visitor_id: str) -> bool:
        return self._reid_gallery.is_reentry(visitor_id)

    def get_billing_queue_depth(self, camera_id: str) -> int:
        return len(self._billing_queue.get(camera_id, set()))

    def update_billing_presence(self, camera_id: str, track_id: int, in_billing: bool):
        if in_billing:
            self._billing_queue[camera_id].add(track_id)
        else:
            self._billing_queue[camera_id].discard(track_id)
