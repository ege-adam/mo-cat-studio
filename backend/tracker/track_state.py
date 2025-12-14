import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from filterpy.kalman import KalmanFilter
from .constraints import MHRJointConstraints

@dataclass
class MHRTrackState:
    """Track state for MHR (Momentum Human Rig) with full body + hands."""
    
    track_id: int
    bbox_kf: KalmanFilter = field(default=None)
    
    joint_history: List[np.ndarray] = field(default_factory=list)
    body_pose_history: List[np.ndarray] = field(default_factory=list)
    hand_pose_history: List[np.ndarray] = field(default_factory=list)
    full_joint_history: List[np.ndarray] = field(default_factory=list)
    full_rot_history: List[np.ndarray] = field(default_factory=list) # Added
    mhr_params_history: List[np.ndarray] = field(default_factory=list)  # For FBX export
    cam_t_history: List[np.ndarray] = field(default_factory=list)  # Root translation for export
    
    shape_params: np.ndarray = field(default=None)
    bone_lengths: Dict[str, float] = field(default=None)
    
    lhand_bbox: np.ndarray = field(default=None)
    rhand_bbox: np.ndarray = field(default=None)
    
    # Raw detection for mesh export
    last_detection: Dict = field(default=None)
    color_hist: np.ndarray = field(default=None)  # Torso histogram
    face_hist: np.ndarray = field(default=None)   # Face region histogram
    
    # Face embedding for re-identification (computed on demand by InsightFace)
    face_embeddings: List[Dict] = field(default_factory=list)
    # Each dict contains:
    # {
    #   'embedding': np.ndarray,      # 512-dim vector
    #   'quality': float,              # 0-1 quality score  
    #   'frame': int,                  # frame captured
    #   'face_crop_b64': str,         # base64 JPEG image
    #   'bbox': [x1, y1, x2, y2],     # face bbox
    # }
    last_embed_frame: int = -1      # Frame when embedding was last computed
    embed_attempts: int = 0          # Number of embedding extraction attempts
    
    # Track lifecycle
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    creation_frame: int = -1         # Frame when track was created
    
    def __post_init__(self):
        if self.bbox_kf is None:
            self._init_bbox_kalman()
    
    def _init_bbox_kalman(self):
        """Initialize Kalman filter for bounding box."""
        self.bbox_kf = KalmanFilter(dim_x=8, dim_z=4)
        
        dt = 1.0
        self.bbox_kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        self.bbox_kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        self.bbox_kf.R *= 10
        self.bbox_kf.Q[-4:, -4:] *= 0.01
        self.bbox_kf.P[4:, 4:] *= 1000
        self.bbox_kf.P *= 10
    
    def initialize(self, detection: Dict):
        """Initialize track from first detection."""
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.bbox_kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0]).reshape(-1, 1)
        
        self.joint_history.append(detection['pred_joint_coords'])
        self.full_joint_history.append(detection['pred_joint_coords']) # Store full history
        self.body_pose_history.append(detection['body_pose_params'])
        self.hand_pose_history.append(detection['hand_pose_params'])
        if 'pred_global_rots' in detection:
             self.full_rot_history.append(detection['pred_global_rots'])
        if 'mhr_model_params' in detection:
             self.mhr_params_history.append(detection['mhr_model_params'])
        if 'pred_cam_t' in detection:
             self.cam_t_history.append(detection['pred_cam_t'])
        
        self.shape_params = detection['shape_params'].copy()
        
        self.lhand_bbox = detection.get('lhand_bbox')
        self.rhand_bbox = detection.get('rhand_bbox')
        
        self.last_detection = detection
        self.color_hist = detection.get('color_hist')
        self.face_hist = detection.get('face_hist')
        
        constraints = MHRJointConstraints()
        self.bone_lengths = constraints.compute_bone_lengths(detection['pred_joint_coords'])
        
        self.hits = 1
    
    def predict(self) -> List[float]:
        """Predict next bbox state."""
        self.bbox_kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()
    
    def update(self, detection: Dict):
        """Update track with new detection."""
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.bbox_kf.update(np.array([cx, cy, w, h]))
        
        self.joint_history.append(detection['pred_joint_coords'])
        self.full_joint_history.append(detection['pred_joint_coords']) # Store full history
        self.body_pose_history.append(detection['body_pose_params'])
        self.hand_pose_history.append(detection['hand_pose_params'])
        if 'pred_global_rots' in detection:
             self.full_rot_history.append(detection['pred_global_rots'])
        if 'mhr_model_params' in detection:
             self.mhr_params_history.append(detection['mhr_model_params'])
        if 'pred_cam_t' in detection:
             self.cam_t_history.append(detection['pred_cam_t'])
        
        max_history = 30
        if len(self.joint_history) > max_history:
            self.joint_history.pop(0)
            self.body_pose_history.pop(0)
            self.hand_pose_history.pop(0)
            # No pop for full_joint_history or full_rot_history as they are meant to be full
        
        if detection.get('shape_params') is not None:
            self.shape_params = 0.95 * self.shape_params + 0.05 * detection['shape_params']
        
        self.lhand_bbox = detection.get('lhand_bbox')
        self.rhand_bbox = detection.get('rhand_bbox')
        
        self.last_detection = detection
        
        # Update color histogram with EMA
        new_hist = detection.get('color_hist')
        if new_hist is not None:
            if self.color_hist is None:
                self.color_hist = new_hist
            else:
                self.color_hist = 0.9 * self.color_hist + 0.1 * new_hist
        
        # Update face histogram with EMA (less aggressive smoothing for faces)
        new_face_hist = detection.get('face_hist')
        if new_face_hist is not None:
            if self.face_hist is None:
                self.face_hist = new_face_hist
            else:
                self.face_hist = 0.85 * self.face_hist + 0.15 * new_face_hist
        
        self.hits += 1
        self.time_since_update = 0
    
    def get_bbox(self) -> List[float]:
        """Get current bbox estimate."""
        state = self.bbox_kf.x.flatten()
        cx, cy, w, h = state[:4]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    def get_smoothed_joints(self, alpha: float = 0.4) -> np.ndarray:
        """Get temporally smoothed joint positions."""
        if len(self.joint_history) < 2:
            return self.joint_history[-1] if self.joint_history else None
        
        constraints = MHRJointConstraints()
        return constraints.smooth_joints_ema(self.joint_history, alpha)
    
    def get_smoothed_poses(self, alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """Get smoothed body and hand poses."""
        def ema_smooth(history, alpha):
            if len(history) < 2:
                return history[-1] if history else None
            
            smoothed = history[-1].copy()
            weight_sum = 1.0
            
            for i in range(1, min(5, len(history))):
                w = alpha * ((1 - alpha) ** i)
                smoothed += w * history[-(i+1)]
                weight_sum += w
            
            return smoothed / weight_sum
        
        body_pose = ema_smooth(self.body_pose_history, alpha)
        hand_pose = ema_smooth(self.hand_pose_history, alpha)
        
        return body_pose, hand_pose
    
    def get_last_cam_t(self) -> Optional[np.ndarray]:
        """Get the last known camera translation (3D root position).
        
        Returns the most recent cam_t from either cam_t_history or last_detection.
        This is used for 3D ghost detection/merging.
        """
        # First try cam_t_history
        if self.cam_t_history and len(self.cam_t_history) > 0:
            return self.cam_t_history[-1]
        
        # Fallback to last_detection
        if self.last_detection is not None:
            cam_t = self.last_detection.get('pred_cam_t')
            if cam_t is not None:
                return np.array(cam_t) if not isinstance(cam_t, np.ndarray) else cam_t
        
        return None
