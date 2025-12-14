import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class CameraTracker:
    """Track camera motion using feature matching on background regions."""
    
    def __init__(self, use_orb: bool = True):
        if use_orb:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.SIFT_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        self.R_accumulated = np.eye(3)
        self.t_accumulated = np.zeros((3, 1))
        self.pose_history = []
    
    def create_background_mask(self,
                               frame_shape: Tuple[int, int],
                               person_bboxes: List[List[float]],
                               margin: int = 20) -> np.ndarray:
        """Create mask for background regions (excluding people)."""
        mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
        
        if person_bboxes:
            for bbox in person_bboxes:
                x1 = max(0, int(bbox[0]) - margin)
                y1 = max(0, int(bbox[1]) - margin)
                x2 = min(frame_shape[1], int(bbox[2]) + margin)
                y2 = min(frame_shape[0], int(bbox[3]) + margin)
                mask[y1:y2, x1:x2] = 0
        
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        return mask
    
    def track(self,
              frame: np.ndarray,
              person_bboxes: List[List[float]] = None,
              camera_matrix: np.ndarray = None) -> Dict:
        """Track camera motion between frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg_mask = self.create_background_mask(frame.shape, person_bboxes or [])
        keypoints, descriptors = self.detector.detectAndCompute(gray, bg_mask)
        
        result = {
            'R': np.eye(3),
            't': np.zeros((3, 1)),
            'R_accumulated': self.R_accumulated.copy(),
            't_accumulated': self.t_accumulated.copy(),
            'num_matches': 0,
            'inlier_ratio': 0.0,
            'is_valid': False
        }
        
        if self.prev_descriptors is not None and descriptors is not None:
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
            
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            result['num_matches'] = len(good_matches)
            
            if len(good_matches) >= 15:
                pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
                
                if camera_matrix is None:
                    h, w = frame.shape[:2]
                    focal = max(h, w)
                    camera_matrix = np.array([
                        [focal, 0, w/2],
                        [0, focal, h/2],
                        [0, 0, 1]
                    ], dtype=np.float32)
                
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, camera_matrix,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                
                if E is not None:
                    inliers = mask.ravel().sum()
                    result['inlier_ratio'] = inliers / len(good_matches)
                    
                    if result['inlier_ratio'] > 0.5:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)
                        
                        result['R'] = R
                        result['t'] = t
                        result['is_valid'] = True
                        
                        self.R_accumulated = R @ self.R_accumulated
                        self.t_accumulated = R @ self.t_accumulated + t
                        
                        result['R_accumulated'] = self.R_accumulated.copy()
                        result['t_accumulated'] = self.t_accumulated.copy()
        
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.pose_history.append(result)
        
        return result
    
    def compensate_bbox(self,
                        bbox: List[float],
                        camera_motion: Dict,
                        depth_estimate: float = 2.0) -> List[float]:
        """Compensate bounding box for camera motion."""
        if not camera_motion['is_valid']:
            return bbox
        
        t = camera_motion['t']
        scale = 500 / depth_estimate
        
        dx = -t[0, 0] * scale
        dy = -t[1, 0] * scale
        
        return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
