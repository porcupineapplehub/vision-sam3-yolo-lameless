"""
Annotation Renderer Service
Renders YOLO bounding boxes and pose keypoints on videos
Like in the research papers: T-LEAP, BiLSTM lameness detection
"""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yaml
import boto3
from botocore.config import Config

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    config=Config(signature_version='s3v4')
)

app = FastAPI(title="Annotation Renderer Service")

# Directories
VIDEOS_DIR = Path("/app/data/videos")
PROCESSED_DIR = Path("/app/data/processed")
RESULTS_DIR = Path("/app/data/results")
ANNOTATED_DIR = PROCESSED_DIR / "annotated"

# Ensure directories exist
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Keypoint names (20 keypoints from Roboflow cow pose dataset)
# CORRECTED mapping based on visual analysis of Roboflow reference annotations
KEYPOINT_NAMES = [
    'left_ear_base',          # 0  - Left ear base / head area
    'neck',                   # 1  - Neck junction
    'withers',                # 2  - Withers (top of shoulders)
    'mid_back',               # 3  - Mid-back (spine)
    'right_hind_hip',         # 4  - Right hind hip
    'right_hind_mid_leg',     # 5  - Right hind mid leg
    'right_hind_fetlock',     # 6  - Right hind fetlock
    'left_hind_shoulder',     # 7  - Left hind hip/shoulder
    'left_hind_mid_leg',      # 8  - Left hind mid leg
    'left_hind_fetlock',      # 9  - Left hind fetlock
    'right_front_shoulder',   # 10 - Right front shoulder
    'right_front_mid_leg',    # 11 - Right front mid leg
    'right_front_lower_leg',  # 12 - Right front lower leg
    'left_front_shoulder',    # 13 - Left front shoulder
    'left_front_mid_leg',     # 14 - Left front mid leg
    'left_front_lower_leg',   # 15 - Left front lower leg
    'right_front_hoof',       # 16 - Right front hoof
    'left_front_hoof',        # 17 - Left front hoof
    'right_hind_hoof',        # 18 - Right hind hoof
    'left_hind_hoof',         # 19 - Left hind hoof (NOT front!)
]

# ============================================================
# Skeleton Structure - CORRECTED from Roboflow reference
# ============================================================
KP_NOSE = 0
KP_NECK = 1
KP_WITHERS = 2
KP_MIDBACK = 3
KP_HIP_L = 7       # left_hind_shoulder
KP_HIP_R = 4       # right_hind_hip

# Fixed chains - CORRECTED based on visual analysis
FR_A = [13, 14, 15, 17]        # LEFT FRONT: left_front_shoulder → hoof
FR_B = [10, 11, 12, 16]        # RIGHT FRONT: right_front_shoulder → hoof
HI_A = [7, 8, 9, 19]           # LEFT HIND: left_hind_shoulder → hoof (includes ID 19!)
HI_B = [4, 5, 6, 18]           # RIGHT HIND: right_hind_hip → hoof

EDGES_BODY = [
    (0, 1),      # ear - neck
    (1, 2),      # neck - withers
    (2, 3),      # withers - mid_back
    (2, 10),     # withers → right_front_shoulder
    (2, 13),     # withers → left_front_shoulder
    (3, 7),      # mid_back → left hind hip
    (3, 4),      # mid_back → right hind hip
]

EDGES_FRONT_A = [(13, 14), (14, 15), (15, 17)]      # LEFT FRONT
EDGES_FRONT_B = [(10, 11), (11, 12), (12, 16)]      # RIGHT FRONT
EDGES_HIND_A = [(7, 8), (8, 9), (9, 19)]            # LEFT HIND
EDGES_HIND_B = [(4, 5), (5, 6), (6, 18)]            # RIGHT HIND

ALL_EDGES: List[Tuple[int, int]] = EDGES_BODY + EDGES_FRONT_A + EDGES_FRONT_B + EDGES_HIND_A + EDGES_HIND_B

# Color scheme for different body parts (BGR format)
SKELETON_COLORS = {
    'head': (0, 255, 255),        # Yellow - head/face
    'spine': (0, 255, 0),         # Green - spine/back line
    'front_left': (255, 0, 0),    # Blue - front left leg
    'front_right': (0, 165, 255), # Orange - front right leg
    'hind_left': (255, 0, 255),   # Magenta - hind left leg
    'hind_right': (255, 255, 0),  # Cyan - hind right leg (FIXED: was yellow)
}

# ------------------------------------------------------------
# Robust keypoint reindexing for mixed sources (YOLO/T-LEAP/etc)
# - Supports: {id,x,y,confidence} OR {name,x,y,confidence}
# - Normalizes name variants
# - Falls back to "already ordered" only if it looks ordered
# ------------------------------------------------------------

NUM_KP = 20

# canonical names must match KEYPOINT_NAMES order exactly
CANON = {n.lower(): i for i, n in enumerate(KEYPOINT_NAMES)}

# optional synonyms/variants you might see in T-LEAP exports
SYN = {
    "headneck": "head_neck",
    "head-neck": "head_neck",
    "midback": "mid_back",
    "mid-back": "mid_back",
    "hipleft": "hip_left",
    "hip-left": "hip_left",
    "hipright": "hip_right",
    "hip-right": "hip_right",
    "frontlshoulder": "front_l_shoulder",
    "frontlelbow": "front_l_elbow",
    "frontlknee": "front_l_knee",
    "frontlhoof": "front_l_hoof",
    "frontrshoulder": "front_r_shoulder",
    "frontrelbow": "front_r_elbow",
    "frontrknee": "front_r_knee",
    "frontrhoof": "front_r_hoof",
    "frontrfetlock": "front_r_fetlock",
    "hindlupper": "hind_l_upper",
    "hindlstifle": "hind_l_stifle",
    "hindlhoof": "hind_l_hoof",
    "hindrstifle": "hind_r_stifle",
    "hindrhoof": "hind_r_hoof",
}

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s2 = s.replace("_", "")
    # map known variants
    if s in SYN:
        return SYN[s].lower()
    if s2 in SYN:
        return SYN[s2].lower()
    return s

def reindex_keypoints_by_id(raw_kps: List[Dict[str, Any]], num_kp: int = NUM_KP) -> List[Dict[str, Any]]:
    """
    Reindex keypoints so that out[i] is the anatomical keypoint with logical id=i.

    CRITICAL RULE FOR ANATOMICAL CORRECTNESS:
    - If 'name' exists, ALWAYS TRUST NAME (canonical anatomical mapping)
    - Only use 'id' field when 'name' is missing
    - If both exist and disagree, log warning and TRUST NAME
    - NEVER guess or fall back to assuming "already ordered"

    This prevents hip_left from ending up at shoulder position (index 13)
    which would cause diagonal cross-body edges.
    """
    out: List[Dict[str, Any]] = [{} for _ in range(num_kp)]
    if not raw_kps:
        return out

    mapped = 0
    for kp in raw_kps:
        if not isinstance(kp, dict):
            continue

        kid = None

        # 1) PREFER NAME mapping if present (anatomically correct)
        if "name" in kp and kp.get("name"):
            nn = _norm_name(kp.get("name"))
            if nn in CANON:
                kid = CANON[nn]
            else:
                # Fallback: try via KEYPOINT_NAMES normalization
                for i, nm in enumerate(KEYPOINT_NAMES):
                    if _norm_name(nm) == nn:
                        kid = i
                        break

            # If also has id, validate consistency (optional but catches bugs)
            if kid is not None and "id" in kp:
                try:
                    raw_id = int(kp["id"])
                    if raw_id != kid:
                        print(f"⚠️  id/name mismatch: name='{kp.get('name')}' maps to {kid}, but id={raw_id}. Trusting name.")
                except Exception:
                    pass

        # 2) ONLY if NAME missing, use ID mapping
        elif "id" in kp:
            try:
                kid = int(kp["id"])
            except Exception:
                kid = None

        if kid is not None and 0 <= kid < num_kp:
            out[kid] = kp
            mapped += 1

    # Strict validation: do NOT guess ordering
    if mapped < 6:
        print(f"⚠️  Insufficient keypoint mapping: {mapped}/{num_kp}. Frame skipped (no guessing).")
        return [{} for _ in range(num_kp)]

    if mapped < 15:
        print(f"⚠️  Partial keypoint mapping: {mapped}/{num_kp} (ok, but may miss some limbs).")

    return out


# ---------- FIX: Robust confidence getter ----------
def _kp_conf(keypoints: List[Dict], idx: int) -> float:
    if idx < 0 or idx >= len(keypoints):
        return 0.0
    kp = keypoints[idx] or {}
    # Accept common variants
    v = kp.get("confidence", None)
    if v is None:
        v = kp.get("conf", None)
    if v is None:
        v = kp.get("score", None)
    if v is None:
        # If your source does not provide confidence at all, treat as visible
        return 1.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _kp_xy(keypoints: List[Dict], idx: int) -> Tuple[Optional[float], Optional[float]]:
    if idx < 0 or idx >= len(keypoints):
        return None, None
    kp = keypoints[idx] or {}
    if "x" not in kp or "y" not in kp:
        return None, None
    return float(kp["x"]), float(kp["y"])


def detect_walking_direction(keypoints: List[Dict]) -> str:
    """
    Per-frame direction:
      RIGHT if kp0.x > kp3.x else LEFT
    Falls back to hips if kp3 is missing.
    """
    x0, _ = _kp_xy(keypoints, KP_NOSE)      # ear/head
    x3, _ = _kp_xy(keypoints, KP_MIDBACK)   # mid_back
    if x0 is not None and x3 is not None:
        return "RIGHT" if x0 > x3 else "LEFT"

    # Fallback: compare ear vs average hips if mid_back missing
    x4, _ = _kp_xy(keypoints, KP_HIP_R)   # right_hind_hip
    x7, _ = _kp_xy(keypoints, KP_HIP_L)   # left_hind_shoulder
    hip_xs = [x for x in [x4, x7] if x is not None]
    if x0 is not None and hip_xs:
        return "RIGHT" if x0 > (sum(hip_xs) / len(hip_xs)) else "LEFT"

    return "LEFT"


def assign_lr_chains() -> Dict[str, List[int]]:
    """
    Return ANATOMICAL leg chains (fixed, no swapping).

    Chains are based on correct Roboflow keypoint order.
    """
    # Always use anatomical identity - NO SWAPPING
    return {
        "FRONT_LEFT_CHAIN": FR_A,    # IDs: 13→14→15→17 (left_front)
        "FRONT_RIGHT_CHAIN": FR_B,   # IDs: 10→11→12→16 (right_front)
        "HIND_LEFT_CHAIN": HI_A,     # IDs: 7→8→9→19 (left_hind)
        "HIND_RIGHT_CHAIN": HI_B,    # IDs: 4→5→6→18 (right_hind)
    }


def edges_from_chain(chain: List[int]) -> List[Tuple[int, int]]:
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


def _normalize_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
    return edge if edge[0] <= edge[1] else (edge[1], edge[0])


def unique_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for e in edges:
        ne = _normalize_edge(e)
        if ne in seen:
            continue
        seen.add(ne)
        out.append(e)
    return out


class RenderRequest(BaseModel):
    video_id: str
    include_yolo: bool = True
    include_pose: bool = True
    show_confidence: bool = False
    show_labels: bool = True
    output_fps: Optional[float] = None
    video_path: Optional[str] = None  # Local path or s3:// URI
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None


# Track rendering progress
render_status: Dict[str, Dict] = {}

# Direction stability tracking (for detecting rapid toggling)
direction_history: Dict[str, List[str]] = {}  # video_id -> list of recent directions


def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: List[Dict],
    bbox: Optional[List[float]] = None,
    confidence_threshold: float = 0.3,
    show_labels: bool = True,
    show_confidence: bool = False
) -> Tuple[np.ndarray, str]:
    """
    Draw cow skeleton on a single frame with direction-aware L/R label assignment.

    Returns:
        Tuple of (annotated_frame, direction) where direction is "LEFT" or "RIGHT"
    """

    # Draw bounding box
    if bbox:
        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(bbox) > 4:  # Has confidence
            cv2.putText(frame, f"Cow {bbox[4]:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not keypoints:
        return frame, "LEFT"

    # T-LEAP outputs keypoints in correct array order (0-19)
    # Array index IS the keypoint ID - use directly, no reindexing needed

    # 1) Determine direction per-frame (user rule) - MUST be on reindexed keypoints
    direction = detect_walking_direction(keypoints)

    # 2) Get anatomical leg chains (no swapping based on direction)
    chains = assign_lr_chains()
    front_left_edges = edges_from_chain(chains["FRONT_LEFT_CHAIN"])
    front_right_edges = edges_from_chain(chains["FRONT_RIGHT_CHAIN"])
    hind_left_edges = edges_from_chain(chains["HIND_LEFT_CHAIN"])
    hind_right_edges = edges_from_chain(chains["HIND_RIGHT_CHAIN"])

    # 3) Build edge list from body + assigned chains only
    # IMPORTANT: Only draw edges from the chains we've assigned, plus body spine
    edges_to_draw = EDGES_BODY + front_left_edges + front_right_edges + hind_left_edges + hind_right_edges

    # Fast membership checks (normalize orientation)
    fl_set = {_normalize_edge(e) for e in front_left_edges}
    fr_set = {_normalize_edge(e) for e in front_right_edges}
    hl_set = {_normalize_edge(e) for e in hind_left_edges}
    hr_set = {_normalize_edge(e) for e in hind_right_edges}
    body_set = {_normalize_edge(e) for e in EDGES_BODY}

    def edge_color(edge: Tuple[int, int]) -> Tuple[int, int, int]:
        ne = _normalize_edge(edge)
        # Priority: label-colored limb edges, then head/spine
        if ne in fl_set:
            return SKELETON_COLORS["front_left"]
        if ne in fr_set:
            return SKELETON_COLORS["front_right"]
        if ne in hl_set:
            return SKELETON_COLORS["hind_left"]
        if ne in hr_set:
            return SKELETON_COLORS["hind_right"]
        if ne == _normalize_edge((0, 1)):
            return SKELETON_COLORS["head"]
        if ne in body_set:
            return SKELETON_COLORS["spine"]
        return (255, 255, 255)

    def can_draw_edge(i: int, j: int) -> bool:
        if i >= len(keypoints) or j >= len(keypoints):
            return False
        if _kp_conf(keypoints, i) < confidence_threshold or _kp_conf(keypoints, j) < confidence_threshold:
            return False
        xi, yi = _kp_xy(keypoints, i)
        xj, yj = _kp_xy(keypoints, j)
        if not (xi is not None and yi is not None and xj is not None and yj is not None):
            return False
        return True

    # Draw all edges (behind keypoints)
    for (i, j) in edges_to_draw:
        if not can_draw_edge(i, j):
            continue
        xi, yi = _kp_xy(keypoints, i)
        xj, yj = _kp_xy(keypoints, j)
        cv2.line(
            frame,
            (int(xi), int(yi)),
            (int(xj), int(yj)),
            edge_color((i, j)),
            3,
        )

    # Keypoint color rule: color by label groups (Front L/R, Hind L/R) + head/spine
    front_left_kps = set(chains["FRONT_LEFT_CHAIN"])
    front_right_kps = set(chains["FRONT_RIGHT_CHAIN"])
    hind_left_kps = set(chains["HIND_LEFT_CHAIN"])
    hind_right_kps = set(chains["HIND_RIGHT_CHAIN"])

    # Draw keypoints on top
    # FIX: Use _kp_conf() consistently (handles confidence/conf/score variants)
    for i in range(len(keypoints)):
        conf = _kp_conf(keypoints, i)

        if conf <= confidence_threshold:
            continue

        x, y = _kp_xy(keypoints, i)
        if x is None or y is None:
            continue

        x, y = int(x), int(y)

        if i in (KP_NOSE, KP_NECK):
            color = SKELETON_COLORS["head"]
        elif i in (KP_WITHERS, KP_MIDBACK, KP_HIP_L, KP_HIP_R):  # spine + hip points
            color = SKELETON_COLORS["spine"]
        elif i in front_left_kps:
            color = SKELETON_COLORS["front_left"]
        elif i in front_right_kps:
            color = SKELETON_COLORS["front_right"]
        elif i in hind_left_kps:
            color = SKELETON_COLORS["hind_left"]
        elif i in hind_right_kps:
            color = SKELETON_COLORS["hind_right"]
        else:
            color = (255, 255, 255)

        # Draw keypoint circle
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White border

        # Draw ID number for ALL keypoints (for mapping verification)
        if show_labels:
            # Draw ID number prominently
            cv2.putText(frame, f"{i}", (x + 10, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{i}", (x + 10, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Optionally show name below (smaller)
            if i in [0, 2, 3, 4, 7, 10, 13]:
                name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else ""
                cv2.putText(frame, name[:8], (x + 10, y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Draw confidence (if enabled)
        if show_confidence:
            cv2.putText(frame, f"{conf:.2f}", (x - 30, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # IMPORTANT: return direction so caller displays the SAME direction used for chain assignment
    return frame, direction


def draw_yolo_on_frame(
    frame: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True
) -> np.ndarray:
    """Draw YOLO detection boxes on frame."""
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        confidence = det.get('confidence', 0)
        class_name = det.get('class', 'cow')
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = class_name
        if show_confidence:
            label = f"{class_name} {confidence:.2f}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def draw_info_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    lameness_score: Optional[float] = None,
    avoid_bbox: Optional[List[float]] = None,
    direction: Optional[str] = None,
) -> np.ndarray:
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Frame info
    time_sec = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if direction:
        cv2.putText(frame, f"Dir: {direction}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    # Lameness score
    if lameness_score is not None:
        score_pct = lameness_score * 100
        if score_pct < 30:
            color = (0, 255, 0)  # Green
            status = "Normal"
        elif score_pct < 60:
            color = (0, 165, 255)  # Orange
            status = "Mild"
        else:
            color = (0, 0, 255)  # Red
            status = "Lame"
        
        cv2.putText(frame, f"Lameness: {score_pct:.1f}% ({status})", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw color legend (anatomical left/right from cow's perspective)
    legend_items = [
        ("Head", SKELETON_COLORS['head']),
        ("Spine", SKELETON_COLORS['spine']),
        ("FL (anat)", SKELETON_COLORS['front_left']),   # Anatomical left
        ("FR (anat)", SKELETON_COLORS['front_right']),  # Anatomical right
        ("HL (anat)", SKELETON_COLORS['hind_left']),    # Anatomical left
        ("HR (anat)", SKELETON_COLORS['hind_right']),   # Anatomical right
    ]

    font_title = 0.42
    font_item = 0.34
    row_h = 14
    pad = 6
    circle_r = 4
    box_w = 115
    box_h = pad + 14 + len(legend_items) * row_h + pad

    def rect_intersection_area(a, b) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    # Candidate placements: top-right, bottom-right, top-left, bottom-left
    margin = 8
    candidates = [
        (w - box_w - margin, margin),              # top-right
        (w - box_w - margin, h - box_h - margin),  # bottom-right
        (margin, margin),                          # top-left
        (margin, h - box_h - margin),              # bottom-left
    ]

    bbox_rect = None
    if avoid_bbox and len(avoid_bbox) >= 4:
        bx1, by1, bx2, by2 = [int(c) for c in avoid_bbox[:4]]
        # expand a bit so legend avoids the cow more aggressively
        expand = 12
        bbox_rect = (max(0, bx1 - expand), max(0, by1 - expand), min(w, bx2 + expand), min(h, by2 + expand))

    best = candidates[0]
    best_overlap = 10**18
    for cx, cy in candidates:
        legend_rect = (cx, cy, cx + box_w, cy + box_h)
        overlap = rect_intersection_area(legend_rect, bbox_rect) if bbox_rect else 0
        if overlap < best_overlap:
            best_overlap = overlap
            best = (cx, cy)

    legend_x, legend_y = best
    x0, y0 = legend_x, legend_y
    x1, y1 = legend_x + box_w, legend_y + box_h
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(frame, "Legend", (x0 + pad, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, font_title, (255, 255, 255), 1)
    
    for i, (name, color) in enumerate(legend_items):
        yy = y0 + 18 + i * row_h
        cv2.circle(frame, (x0 + pad + circle_r, yy - 4), circle_r, color, -1)
        cv2.putText(frame, name, (x0 + pad + 2 * circle_r + 6, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_item, (255, 255, 255), 1)
    
    return frame


async def render_annotated_video(request: RenderRequest):
    """Render annotated video with pose and/or YOLO detections."""
    video_id = request.video_id
    temp_video_file = None  # Track temp file for cleanup

    # Update status
    render_status[video_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Loading data...'
    }

    try:
        # Find video file - check S3 first, then local
        video_path = None

        if request.s3_bucket and request.s3_key:
            # Download video from S3
            render_status[video_id] = {
                'status': 'starting',
                'progress': 0,
                'message': 'Downloading video from S3...'
            }
            print(f"📥 Downloading video from S3: s3://{request.s3_bucket}/{request.s3_key}")

            # Get the file extension
            ext = Path(request.s3_key).suffix or '.mp4'
            temp_video_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            temp_video_file.close()

            try:
                s3_client.download_file(request.s3_bucket, request.s3_key, temp_video_file.name)
                video_path = Path(temp_video_file.name)
                print(f"✅ Downloaded video to {temp_video_file.name}")
            except Exception as e:
                print(f"❌ Failed to download from S3: {e}")
                render_status[video_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'Failed to download video from S3: {e}'
                }
                return
        else:
            # Check local filesystem
            video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
            if not video_files:
                render_status[video_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': 'Video file not found'
                }
                return
            video_path = video_files[0]
        
        # Load YOLO results
        yolo_data = None
        if request.include_yolo:
            yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
            if yolo_file.exists():
                with open(yolo_file) as f:
                    yolo_data = json.load(f)
        
        # Load T-LEAP pose results
        pose_data = None
        lameness_score = None
        if request.include_pose:
            pose_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
            if pose_file.exists():
                with open(pose_file) as f:
                    pose_data = json.load(f)
                    lameness_score = pose_data.get('locomotion_features', {}).get('lameness_score')
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            render_status[video_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to open video'
            }
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_fps = request.output_fps or fps
        
        # Create output video
        output_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        # Create frame lookup from pose data
        pose_by_frame = {}
        if pose_data and 'pose_sequences' in pose_data:
            for seq in pose_data['pose_sequences']:
                frame_idx = seq.get('frame', 0)
                pose_by_frame[frame_idx] = seq
        
        # Create frame lookup from YOLO data
        yolo_by_frame = {}
        if yolo_data and 'detections' in yolo_data:
            for det in yolo_data['detections']:
                if isinstance(det, dict):
                    frame_idx = det.get('frame', 0)
                    yolo_by_frame[frame_idx] = det.get('detections', [])
        
        render_status[video_id] = {
            'status': 'rendering',
            'progress': 0,
            'message': 'Rendering frames...'
        }

        # Initialize direction tracking for this video
        direction_history[video_id] = []
        direction_flip_count = 0
        last_direction = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw YOLO detections
            if request.include_yolo and frame_idx in yolo_by_frame:
                frame = draw_yolo_on_frame(frame, yolo_by_frame[frame_idx], 
                                          request.show_confidence)
            
            # Draw pose skeleton
            direction = None
            bbox = None
            if request.include_pose and frame_idx in pose_by_frame:
                pose_seq = pose_by_frame[frame_idx]
                keypoints = pose_seq.get('keypoints', [])
                bbox = pose_seq.get('bbox')

                # Add confidence to bbox if available
                if bbox and pose_seq.get('detection_confidence'):
                    bbox = bbox + [pose_seq['detection_confidence']]

                # FIX: draw_skeleton_on_frame now returns (frame, direction)
                # This ensures the direction shown in overlay matches the one used for chain assignment
                frame, direction = draw_skeleton_on_frame(
                    frame, keypoints, bbox,
                    show_labels=request.show_labels,
                    show_confidence=request.show_confidence
                )

                # Track direction stability (detect rapid flipping)
                if last_direction and direction != last_direction:
                    direction_flip_count += 1
                    if direction_flip_count == 5:  # Warn after 5 flips
                        print(f"⚠️  Direction toggling detected in video {video_id} (frame {frame_idx})")
                        print(f"    Consider adding direction hysteresis/smoothing")
                last_direction = direction

            # Draw info overlay (pass cow bbox so legend can avoid it)
            frame = draw_info_overlay(frame, frame_idx, fps, lameness_score, avoid_bbox=bbox, direction=direction)
            
            out.write(frame)
            frame_idx += 1
            
            # Update progress every 5 frames for smoother progress display
            if frame_idx % 5 == 0 or frame_idx == 1:
                progress = (frame_idx / total_frames) * 100
                render_status[video_id] = {
                    'status': 'rendering',
                    'progress': round(progress, 1),
                    'message': f'Rendering frame {frame_idx}/{total_frames}'
                }
        
        cap.release()
        out.release()

        # Print diagnostic summary
        print(f"\n{'='*60}")
        print(f"Rendering complete for video: {video_id}")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_idx}")
        print(f"Direction flips detected: {direction_flip_count}")
        if direction_flip_count > frame_idx / 10:
            print(f"⚠️  High flip rate ({direction_flip_count}/{frame_idx}) - consider direction smoothing")
        print(f"{'='*60}\n")

        # Clean up direction history
        if video_id in direction_history:
            del direction_history[video_id]

        # Convert to web-compatible format using ffmpeg if available
        try:
            import subprocess
            web_output = ANNOTATED_DIR / f"{video_id}_annotated_web.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', str(output_path),
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-movflags', '+faststart',
                str(web_output)
            ], capture_output=True, check=True)
            # Replace with web-compatible version
            output_path.unlink()
            web_output.rename(output_path)
        except Exception:
            pass  # Keep original if ffmpeg fails
        
        render_status[video_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Rendering complete',
            'output_path': str(output_path)
        }
        
        print(f"✅ Rendered annotated video: {output_path}")

    except Exception as e:
        print(f"❌ Error rendering video {video_id}: {e}")
        import traceback
        traceback.print_exc()
        render_status[video_id] = {
            'status': 'error',
            'progress': 0,
            'message': str(e)
        }

    finally:
        # Clean up temp file if we downloaded from S3
        if temp_video_file and os.path.exists(temp_video_file.name):
            try:
                os.unlink(temp_video_file.name)
                print(f"🗑️ Cleaned up temp file: {temp_video_file.name}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temp file: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/render")
async def start_render(request: RenderRequest, background_tasks: BackgroundTasks):
    """Start rendering an annotated video."""
    video_id = request.video_id
    
    # Check if already rendering
    if video_id in render_status and render_status[video_id].get('status') == 'rendering':
        return {"status": "already_rendering", "video_id": video_id}
    
    # Start background rendering
    background_tasks.add_task(render_annotated_video, request)
    
    return {"status": "started", "video_id": video_id}


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Get rendering status for a video."""
    if video_id not in render_status:
        # Check if annotated file exists
        output_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if output_path.exists():
            return {
                "status": "complete",
                "progress": 100,
                "message": "Annotated video available",
                "output_path": str(output_path)
            }
        return {"status": "not_found", "progress": 0, "message": "No render status found"}
    
    return render_status[video_id]


@app.delete("/status/{video_id}")
async def clear_status(video_id: str):
    """Clear rendering status for a video."""
    if video_id in render_status:
        del render_status[video_id]
    return {"status": "cleared", "video_id": video_id}


@app.get("/")
async def root():
    return {
        "service": "Annotation Renderer",
        "version": "1.0",
        "keypoints": len(KEYPOINT_NAMES),
        "skeleton_connections": len(unique_edges(ALL_EDGES)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
