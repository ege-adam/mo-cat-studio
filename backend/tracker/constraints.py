import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

class MHRJointConstraints:
    """
    Anatomical constraints for Momentum Human Rig (MHR).
    127 joints including body, hands, and face.
    """
    
    NUM_JOINTS = 127
    NUM_BODY_POSE_PARAMS = 133
    NUM_HAND_POSE_PARAMS = 108  # 54 per hand
    NUM_EXPR_PARAMS = 72
    
    MAX_ANGULAR_VELOCITY = {
        'body': 0.5,
        'hand': 0.8,
        'finger': 1.0,
        'face': 0.6,
        'spine': 0.3,
        'head': 0.4,
        'default': 0.5,
    }
    
    def __init__(self, use_gpu=False):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    def clamp_pose_params(self, 
                          body_pose: np.ndarray,
                          hand_pose: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply soft clamping to pose parameters."""
        body_pose = np.clip(body_pose, -np.pi, np.pi)
        
        if hand_pose is not None:
            hand_pose = np.clip(hand_pose, -np.pi/2, np.pi/2)
        
        return body_pose, hand_pose
    
    def enforce_velocity_limits(self,
                                current_joints: np.ndarray,
                                previous_joints: np.ndarray,
                                dt: float = 1/30) -> np.ndarray:
        """Limit joint velocities for temporal smoothness."""
        if previous_joints is None:
            return current_joints
        
        max_velocity = 0.5  # meters per frame at 30fps
        
        delta = current_joints - previous_joints
        velocities = np.linalg.norm(delta, axis=1)
        
        too_fast = velocities > max_velocity
        
        if np.any(too_fast):
            scale = np.ones(len(velocities))
            scale[too_fast] = max_velocity / velocities[too_fast]
            delta = delta * scale[:, np.newaxis]
            current_joints = previous_joints + delta
        
        return current_joints
    
    def smooth_joints_ema(self,
                          joint_history: List[np.ndarray],
                          alpha: float = 0.4) -> np.ndarray:
        """Exponential moving average smoothing for joints."""
        if len(joint_history) < 2:
            return joint_history[-1] if joint_history else None
        
        smoothed = joint_history[-1].copy()
        weight_sum = 1.0
        
        for i in range(1, min(5, len(joint_history))):
            w = alpha * ((1 - alpha) ** i)
            smoothed += w * joint_history[-(i+1)]
            weight_sum += w
        
        return smoothed / weight_sum
    
    def compute_bone_lengths(self, joints_3d: np.ndarray) -> Dict[str, float]:
        """Compute bone lengths from joint positions."""
        bones = {
            'spine': (0, 3),
            'neck': (3, 12),
            'left_upper_arm': (12, 14),
            'left_lower_arm': (14, 16),
            'right_upper_arm': (13, 15),
            'right_lower_arm': (15, 17),
            'left_upper_leg': (1, 4),
            'left_lower_leg': (4, 7),
            'right_upper_leg': (2, 5),
            'right_lower_leg': (5, 8),
        }
        
        lengths = {}
        for name, (i, j) in bones.items():
            if i < len(joints_3d) and j < len(joints_3d):
                lengths[name] = np.linalg.norm(joints_3d[j] - joints_3d[i])
        
        return lengths
