import cv2
import numpy as np
from typing import List, Dict


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def compute_joint_distance(joints1: np.ndarray, joints2: np.ndarray) -> float:
    """Compute normalized distance between joint sets."""
    if joints1 is None or joints2 is None:
        return 1.0

    # Use subset of key joints for speed
    key_joints = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

    j1 = joints1[key_joints] if len(joints1) > max(key_joints) else joints1[:10]
    j2 = joints2[key_joints] if len(joints2) > max(key_joints) else joints2[:10]

    # Normalize by root
    j1 = j1 - j1[0:1]
    j2 = j2 - j2[0:1]

    dist = np.mean(np.linalg.norm(j1 - j2, axis=1))
    return min(dist / 0.5, 1.0)


def compute_face_hist(image: np.ndarray, joints: np.ndarray, bbox: List[float]) -> np.ndarray:
    """Compute HSV color histogram for face region using MHR keypoints.

    Uses face keypoints (nose, eyes, ears) to estimate face bounding box.
    MHR 70 keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear, 69=neck

    Args:
        image: BGR image
        joints: 2D joint coordinates (N, 2) or (N, 3) - using first 5 for face
        bbox: Person bounding box [x1, y1, x2, y2]

    Returns:
        Face histogram (32 bins for H channel)
    """
    if joints is None or len(joints) < 5:
        return np.zeros(32, dtype=np.float32)

    # Face keypoint indices: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)
    face_indices = [0, 1, 2, 3, 4]
    face_joints = joints[face_indices, :2] if joints.shape[1] >= 2 else joints[face_indices]

    # Get face bounding box from keypoints
    valid_points = face_joints[~np.any(np.isnan(face_joints), axis=1)]
    if len(valid_points) < 2:
        return np.zeros(32, dtype=np.float32)

    face_x_min = np.min(valid_points[:, 0])
    face_x_max = np.max(valid_points[:, 0])
    face_y_min = np.min(valid_points[:, 1])
    face_y_max = np.max(valid_points[:, 1])

    # Expand the face box by 30% for better coverage
    width = face_x_max - face_x_min
    height = face_y_max - face_y_min

    # Minimum size check
    if width < 10 or height < 10:
        # Fall back to upper 20% of body bbox as face region
        x1, y1, x2, y2 = map(int, bbox)
        face_x_min = x1 + (x2 - x1) * 0.2
        face_x_max = x2 - (x2 - x1) * 0.2
        face_y_min = y1
        face_y_max = y1 + (y2 - y1) * 0.2
        width = face_x_max - face_x_min
        height = face_y_max - face_y_min

    margin = 0.3
    face_x_min -= width * margin
    face_x_max += width * margin
    face_y_min -= height * margin
    face_y_max += height * margin

    # Clip to image bounds
    h, w = image.shape[:2]
    face_x_min = max(0, int(face_x_min))
    face_x_max = min(w, int(face_x_max))
    face_y_min = max(0, int(face_y_min))
    face_y_max = min(h, int(face_y_max))

    if face_x_max <= face_x_min or face_y_max <= face_y_min:
        return np.zeros(32, dtype=np.float32)

    # Crop face region
    face_crop = image[face_y_min:face_y_max, face_x_min:face_x_max]

    if face_crop.size == 0:
        return np.zeros(32, dtype=np.float32)

    # Convert to HSV and compute histogram (focus on Hue for skin tone)
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

    # Compute Hue histogram (32 bins) - skin tone is most distinctive
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_h.flatten()


def compute_color_hist(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """Compute HSV color histogram for upper-body region of the bbox.

    Uses upper 60% of bbox (torso area) to avoid legs which can vary more.
    Includes H, S, V channels for better discrimination.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(64, dtype=np.float32)  # Return empty hist (32H + 16S + 16V = 64)

    # Use upper 60% of bbox (torso/upper body) for more stable color
    bbox_height = y2 - y1
    upper_y2 = y1 + int(bbox_height * 0.6)

    # Also shrink horizontally by 10% on each side to reduce background contamination
    bbox_width = x2 - x1
    inner_x1 = x1 + int(bbox_width * 0.1)
    inner_x2 = x2 - int(bbox_width * 0.1)

    if inner_x2 <= inner_x1 or upper_y2 <= y1:
        return np.zeros(64, dtype=np.float32)

    crop = image[y1:upper_y2, inner_x1:inner_x2]

    if crop.size == 0:
        return np.zeros(64, dtype=np.float32)

    # Convert to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Compute histogram with H, S, V channels for better discrimination
    # H: 32 bins (more granular hue), S: 16 bins, V: 16 bins -> 64 features total
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return hist


def compute_hist_dist(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute correlation distance between histograms (1 - correlation)."""
    if hist1 is None or hist2 is None:
        return 1.0
    if len(hist1) == 0 or len(hist2) == 0:
        return 1.0
    if np.sum(hist1) == 0 or np.sum(hist2) == 0:
        return 1.0
    # Correlation: 1 is perfect match, -1 is mismatch.
    # We want distance: 0 is perfect match.
    score = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    return 1.0 - max(0, score) # Clamp to [0, 1] roughly


def compute_combined_appearance_dist(track, det: Dict) -> float:
    """Compute combined appearance distance using face and torso histograms.

    Face histogram is weighted higher when available as it's more distinctive.

    Args:
        track: Track object with color_hist and face_hist attributes
        det: Detection dict with 'color_hist' and 'face_hist' keys

    Returns:
        Combined distance in [0, 1] where 0 is perfect match
    """
    # Torso (body) histogram distance
    torso_dist = compute_hist_dist(track.color_hist, det.get('color_hist'))

    # Face histogram distance
    face_dist = compute_hist_dist(track.face_hist, det.get('face_hist'))

    # Check if face histograms are available and valid
    track_has_face = track.face_hist is not None and np.sum(track.face_hist) > 0
    det_has_face = det.get('face_hist') is not None and np.sum(det.get('face_hist', np.array([]))) > 0

    if track_has_face and det_has_face:
        # Both have face: weight face higher (60% face, 40% torso)
        return 0.6 * face_dist + 0.4 * torso_dist
    else:
        # Fall back to torso only
        return torso_dist
