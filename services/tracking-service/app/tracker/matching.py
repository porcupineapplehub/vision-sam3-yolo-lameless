"""
Matching utilities for multi-object tracking.

Implements IoU matching, appearance matching, and the Hungarian algorithm
for optimal assignment between detections and tracks.
"""
import numpy as np
from typing import List, Tuple, Optional
import lap


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_test: (N, 4) array of [x1, y1, x2, y2] boxes
        bb_gt: (M, 4) array of [x1, y1, x2, y2] boxes

    Returns:
        (N, M) IoU matrix
    """
    bb_test = np.atleast_2d(bb_test)
    bb_gt = np.atleast_2d(bb_gt)

    # Expand dimensions for broadcasting
    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0:1].T)
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1:2].T)
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2:3].T)
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3:4].T)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    intersection = w * h

    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    union = area_test[:, np.newaxis] + area_gt[np.newaxis, :] - intersection

    iou = intersection / (union + 1e-6)

    return iou


def cosine_distance(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance matrix between two feature sets.

    Args:
        features1: (N, D) feature matrix
        features2: (M, D) feature matrix

    Returns:
        (N, M) distance matrix (1 - cosine_similarity)
    """
    # Normalize features
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-6)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-6)

    # Compute cosine similarity
    similarity = features1 @ features2.T

    # Convert to distance
    distance = 1.0 - similarity

    return distance


def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve linear assignment problem using the Hungarian algorithm.

    Args:
        cost_matrix: (N, M) cost matrix

    Returns:
        matched_indices: (K, 2) array of matched (row, col) pairs
        unmatched_rows: Indices of unmatched rows
        unmatched_cols: Indices of unmatched columns
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1])
        )

    # Use LAP (Linear Assignment Problem) solver
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=100000)

    matched_indices = []
    for i, j in enumerate(x):
        if j >= 0:
            matched_indices.append([i, j])

    matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int)

    unmatched_rows = np.array([i for i, j in enumerate(x) if j < 0])
    unmatched_cols = np.array([j for j, i in enumerate(y) if i < 0])

    return matched_indices, unmatched_rows, unmatched_cols


def associate_detections_to_tracks(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3,
    detection_features: Optional[np.ndarray] = None,
    track_features: Optional[np.ndarray] = None,
    appearance_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to existing tracks using IoU and optionally appearance.

    Args:
        detections: (N, 4) array of detection bboxes
        tracks: (M, 4) array of track bboxes
        iou_threshold: Minimum IoU for valid match
        detection_features: (N, D) detection embeddings (optional)
        track_features: (M, D) track embeddings (optional)
        appearance_weight: Weight of appearance vs IoU (0-1)

    Returns:
        matched: (K, 2) array of matched (detection_idx, track_idx) pairs
        unmatched_detections: Indices of unmatched detections
        unmatched_tracks: Indices of unmatched tracks
    """
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty(0, dtype=int)
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(len(tracks))
        )

    # Compute IoU matrix
    iou_matrix = iou_batch(detections, tracks)

    # Combine with appearance if available
    if detection_features is not None and track_features is not None:
        appearance_dist = cosine_distance(detection_features, track_features)
        # Convert IoU to distance (higher is better -> lower distance)
        iou_dist = 1.0 - iou_matrix
        # Weighted combination
        cost_matrix = (1 - appearance_weight) * iou_dist + appearance_weight * appearance_dist
    else:
        # Just use IoU as cost (inverted)
        cost_matrix = 1.0 - iou_matrix

    # Solve assignment
    matched, unmatched_detections, unmatched_tracks = linear_assignment(cost_matrix)

    # Filter matches below IoU threshold
    valid_matches = []
    for m in matched:
        if iou_matrix[int(m[0]), int(m[1])] >= iou_threshold:
            valid_matches.append([int(m[0]), int(m[1])])
        else:
            unmatched_detections = np.append(unmatched_detections, int(m[0]))
            unmatched_tracks = np.append(unmatched_tracks, int(m[1]))

    matched = np.array(valid_matches, dtype=int) if valid_matches else np.empty((0, 2), dtype=int)
    unmatched_detections = unmatched_detections.astype(int)
    unmatched_tracks = unmatched_tracks.astype(int)

    return matched, unmatched_detections, unmatched_tracks


def fuse_scores(detection_scores: np.ndarray, iou_matrix: np.ndarray) -> np.ndarray:
    """
    Fuse detection scores with track overlap for ByteTrack-style association.

    Higher scores indicate better matches.

    Args:
        detection_scores: (N,) confidence scores
        iou_matrix: (N, M) IoU matrix

    Returns:
        (N, M) fused score matrix
    """
    # Scale IoU by detection confidence
    fused = iou_matrix * detection_scores[:, np.newaxis]
    return fused
