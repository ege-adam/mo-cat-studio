import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from .constraints import MHRJointConstraints
from .camera import CameraTracker
from .track_state import MHRTrackState
from .face_recognizer import get_face_recognizer
from . import similarity_metrics
from .face_embedding_manager import FaceEmbeddingManager
from .track_merger import TrackMerger

class MHRVideoTracker:
    """
    Multi-person video tracker for SAM 3D Body / MHR output.
    
    Features:
    - Kalman filter based bbox prediction
    - Hungarian algorithm matching
    - Camera motion compensation
    - Temporal pose smoothing
    - Full body + hands + face tracking (127 joints)
    """
    
    def __init__(self, estimator,
                 use_gpu: bool = False,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 pose_sim_weight: float = 0.3,
                 ghosting_threshold: float = 0.25,  # Lowered from 0.4 to 0.25 for tighter ghost detection
                 bbox_threshold: float = 0.5,
                 use_fp16: bool = False,
                 disable_ghost_suppression: bool = False,
                 ghost_3d_threshold: float = 0.5,  # 3D distance threshold in meters for ghost merging
                 hist_match_threshold: float = 0.3,  # Histogram similarity threshold for ghost merging
                 max_embeddings_per_person: int = 10,
                 embed_similarity_threshold: float = 0.9,
                 distance_limit: float = 20.0,
                 use_3d_matching: bool = True,  # Enable 3D distance in active track matching
                 dist_3d_weight: float = 0.2,  # Weight for 3D distance in cost matrix
                 embed_retry_interval: int = 10,  # Frames between face embedding retry attempts
                 embed_max_retries: int = 3,  # Maximum face embedding retry attempts
                 embed_quality_threshold: float = 0.4,  # Minimum quality for face embedding
                 gallery_max_age: int = 300,  # Frames to keep lost tracks in gallery
                 face_match_threshold: float = 0.5):  # Cosine similarity threshold for face matching
        self.estimator = estimator
        self.joint_constraints = MHRJointConstraints(use_gpu)
        self.camera_tracker = CameraTracker()
        
        self.tracks: Dict[int, MHRTrackState] = {}
        self.next_id = 0

        # Face recognizer (lazy loaded)
        self._face_recognizer = None
        
        # Parameters
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.pose_sim_weight = pose_sim_weight
        self.ghosting_threshold = ghosting_threshold
        self.bbox_threshold = bbox_threshold
        self.use_fp16 = use_fp16
        self.disable_ghost_suppression = disable_ghost_suppression
        self.ghost_3d_threshold = ghost_3d_threshold
        self.hist_match_threshold = hist_match_threshold
        self.distance_limit = distance_limit
        self.use_3d_matching = use_3d_matching
        self.dist_3d_weight = dist_3d_weight

        # Store merge events for frontend notification
        self.merge_events = []

        # Initialize FaceEmbeddingManager
        self.face_embedding_manager = FaceEmbeddingManager(
            face_recognizer=self.face_recognizer,
            embed_retry_interval=embed_retry_interval,
            embed_max_retries=embed_max_retries,
            embed_quality_threshold=embed_quality_threshold,
            gallery_max_age=gallery_max_age,
            face_match_threshold=face_match_threshold,
            max_embeddings_per_person=max_embeddings_per_person,
            embed_similarity_threshold=embed_similarity_threshold,
            distance_limit=distance_limit
        )

        # Initialize TrackMerger
        self.track_merger = TrackMerger(
            face_recognizer=self.face_recognizer,
            disable_ghost_suppression=disable_ghost_suppression,
            ghost_3d_threshold=ghost_3d_threshold,
            hist_match_threshold=hist_match_threshold,
            max_embeddings_per_person=max_embeddings_per_person,
            embed_similarity_threshold=embed_similarity_threshold,
            min_hits=min_hits
        )

        print(f"MHRVideoTracker initialized with ghosting_threshold={self.ghosting_threshold}, max_age={self.max_age}, bbox_threshold={self.bbox_threshold}, use_fp16={self.use_fp16}, disable_ghost_suppression={self.disable_ghost_suppression}, ghost_3d_threshold={self.ghost_3d_threshold}, hist_match_threshold={self.hist_match_threshold}")
    
    @property
    def face_recognizer(self):
        """Lazy-load face recognizer."""
        if self._face_recognizer is None:
            self._face_recognizer = get_face_recognizer(device='cuda')
        return self._face_recognizer
    

    
    
    
    

    
    def match_detections_to_tracks(self,
                                   detections: List[Dict],
                                   camera_motion: Dict) -> Tuple[Dict, List, List]:
        """Match detections to tracks using Hungarian algorithm."""
        if not self.tracks:
            return {}, [], list(range(len(detections)))
        
        if not detections:
            return {}, list(self.tracks.keys()), []
        
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)
        
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            pred_bbox = track.predict()
            pred_bbox = self.camera_tracker.compensate_bbox(pred_bbox, camera_motion)
            prev_joints = track.get_smoothed_joints()
            
            for j, det in enumerate(detections):
                iou = similarity_metrics.compute_iou(pred_bbox, det['bbox'])
                joint_dist = similarity_metrics.compute_joint_distance(
                    prev_joints, det['pred_joint_coords']
                )
                # Use combined appearance distance (face + torso)
                hist_dist = similarity_metrics.compute_combined_appearance_dist(track, det)

                # Get 3D positions for 3D distance computation
                dist_3d = 0.0
                has_3d = False
                if self.use_3d_matching:
                    track_cam_t = track.get_last_cam_t()
                    det_cam_t = det.get('pred_cam_t')

                    if track_cam_t is not None and det_cam_t is not None:
                        dist_3d_raw = np.linalg.norm(np.array(track_cam_t) - np.array(det_cam_t))
                        # Normalize by threshold to get 0-1 range
                        dist_3d = min(dist_3d_raw / self.ghost_3d_threshold, 1.0)
                        has_3d = True

                # Adaptive Weights based on track status and 3D availability
                if has_3d:
                    # Use 3D-enhanced weights
                    if track.time_since_update > 0:
                        # LOST TRACK RECOVERY MODE with 3D
                        # Prioritize 3D + Color for recovery
                        color_weight = 0.5
                        pose_weight = 0.2
                        iou_weight = 0.05
                        dist_3d_weight = 0.25
                    else:
                        # NORMAL TRACKING MODE with 3D
                        # Balanced weights
                        color_weight = 0.30
                        pose_weight = 0.25
                        iou_weight = 0.25
                        dist_3d_weight = self.dist_3d_weight  # Default 0.20
                    expansion_factor = 1.0 + (track.time_since_update * 0.1) if track.time_since_update > 0 else 1.0
                else:
                    # Fallback to original weights without 3D
                    if track.time_since_update > 0:
                        # LOST TRACK RECOVERY MODE
                        # If track is lost, trust Color more, IoU less (KF drifts)
                        color_weight = 0.6
                        pose_weight = 0.3
                        iou_weight = 0.1
                        expansion_factor = 1.0 + (track.time_since_update * 0.1)
                    else:
                        # NORMAL TRACKING MODE
                        # Weights: IoU=0.3, Pose=0.3, Color=0.4
                        color_weight = 0.4
                        pose_weight = self.pose_sim_weight  # 0.3
                        iou_weight = 1.0 - pose_weight - color_weight
                        expansion_factor = 1.0

                # Spatial Gating: If distance is too large, increase cost significantly
                # Calculate center distance relative to bbox size
                pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
                pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
                det_cx = (det['bbox'][0] + det['bbox'][2]) / 2
                det_cy = (det['bbox'][1] + det['bbox'][3]) / 2
                
                center_dist = np.sqrt((pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)
                avg_size = ((pred_bbox[2] - pred_bbox[0]) + (pred_bbox[3] - pred_bbox[1])) / 2
                
                # Dynamic Search Radius
                search_radius = 2.0 * avg_size * expansion_factor
                
                # If center distance is > search_radius and IoU is very low, it's likely a teleportation (unless IoU > 0)
                spatial_penalty = 0.0
                if center_dist > search_radius and iou <= 0.01:
                    spatial_penalty = 1.0 # Huge penalty
                
                # Track confirmation penalty: new tracks (low hits) get a penalty
                # This prevents new tracks from stealing detections from established tracks
                # Penalty decreases as track gets more hits (more confirmed)
                confirmation_penalty = 0.0
                if track.hits < self.min_hits:
                    # New track: add penalty inversely proportional to hits
                    # hits=1 -> penalty=0.15, hits=2 -> penalty=0.10, hits=3 -> penalty=0.05
                    confirmation_penalty = 0.05 * (self.min_hits - track.hits)

                # Compute final cost with or without 3D distance
                if has_3d:
                    cost = iou_weight * (1 - iou) + \
                           pose_weight * joint_dist + \
                           color_weight * hist_dist + \
                           dist_3d_weight * dist_3d + \
                           spatial_penalty + \
                           confirmation_penalty
                else:
                    cost = iou_weight * (1 - iou) + \
                           pose_weight * joint_dist + \
                           color_weight * hist_dist + \
                           spatial_penalty + \
                           confirmation_penalty
                cost_matrix[i, j] = cost
        
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        matches = {}
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(n_dets))
        
        # Store IoU values for match validation
        iou_matrix = np.zeros((n_tracks, n_dets))
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            pred_bbox = track.get_bbox()  # Use current bbox estimate
            for j, det in enumerate(detections):
                iou_matrix[i, j] = similarity_metrics.compute_iou(pred_bbox, det['bbox'])
        
        for row, col in zip(row_idx, col_idx):
            track_id = track_ids[row]
            track = self.tracks[track_id]
            iou_val = iou_matrix[row, col]
            cost_val = cost_matrix[row, col]
            
            # Stricter matching: lower cost threshold (0.5 instead of 0.7)
            # Also require minimum IoU for established tracks (prevents ID switches)
            # For lost tracks (time_since_update > 0), relax IoU requirement
            if track.time_since_update > 0:
                # Lost track recovery: trust cost more, IoU can be low
                if cost_val < 0.6:
                    matches[track_id] = col
                    unmatched_tracks.discard(track_id)
                    unmatched_dets.discard(col)
            else:
                # Active track: require good cost AND some spatial overlap
                # IoU > 0.05 means at least some overlap (not completely disjoint)
                if cost_val < 0.5 and iou_val > 0.05:
                    matches[track_id] = col
                    unmatched_tracks.discard(track_id)
                    unmatched_dets.discard(col)
        
        # Secondary matching: Distance based for "ghosting" prevention
        # If a new detection is very close to an unmatched track, force match
        # This handles cases where IoU/Pose failed but spatial location is consistent
        
        # Convert sets to lists for iteration
        rem_tracks = list(unmatched_tracks)
        rem_dets = list(unmatched_dets)
        
        for track_id in rem_tracks:
            if not rem_dets:
                break
                
            track = self.tracks[track_id]
            pred_bbox = track.predict()
            pred_bbox = self.camera_tracker.compensate_bbox(pred_bbox, camera_motion)
            
            # Calculate center of predicted bbox
            pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
            pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
            
            best_det_idx = -1
            min_dist = float('inf')
            
            # Find closest unmatched detection
            for det_idx in rem_dets:
                det_bbox = detections[det_idx]['bbox']
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                
                # Euclidean distance between centers (normalized by image size roughly? 
                # BBox coords are usually pixels. Let's assume pixels.)
                dist = np.sqrt((pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)
                
                # Threshold: e.g., 50 pixels or relative to bbox size
                # Let's use relative to bbox width/height average
                avg_size = ((pred_bbox[2] - pred_bbox[0]) + (pred_bbox[3] - pred_bbox[1])) / 2
                threshold = avg_size * self.ghosting_threshold # Allow movement up to X% of body size
                
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    best_det_idx = det_idx
            
            if best_det_idx != -1:
                matches[track_id] = best_det_idx
                unmatched_tracks.discard(track_id)
                unmatched_dets.discard(best_det_idx)
                rem_dets.remove(best_det_idx) # Remove from local list to avoid double matching
        
        # ---------------------------------------------------------
        # Ghost Suppression:
        # Check if remaining unmatched detections are actually ghosts of matched tracks.
        # If an unmatched detection is very close to a matched detection/track, discard it.
        # ---------------------------------------------------------
        if self.disable_ghost_suppression:
            # Skip ghost suppression if disabled
            final_unmatched_dets = list(unmatched_dets)
        else:
            final_unmatched_dets = []
            for det_idx in unmatched_dets:
                det_bbox = detections[det_idx]['bbox']
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                det_size = ((det_bbox[2] - det_bbox[0]) + (det_bbox[3] - det_bbox[1])) / 2
                
                is_ghost = False
                
                # Check against all matched detections
                for track_id, match_idx in matches.items():
                    matched_bbox = detections[match_idx]['bbox']
                    matched_cx = (matched_bbox[0] + matched_bbox[2]) / 2
                    matched_cy = (matched_bbox[1] + matched_bbox[3]) / 2
                    
                    dist = np.sqrt((det_cx - matched_cx)**2 + (det_cy - matched_cy)**2)
                    threshold = det_size * self.ghosting_threshold
                    
                    if dist < threshold:
                        # Check pose similarity to confirm it's a ghost
                        # If poses are different, it's likely a different person (occlusion)
                        joint_dist = similarity_metrics.compute_joint_distance(
                            detections[det_idx]['pred_joint_coords'],
                            detections[match_idx]['pred_joint_coords']
                        )

                        # Check color similarity (torso)
                        torso_dist = similarity_metrics.compute_hist_dist(
                            detections[det_idx].get('color_hist'),
                            detections[match_idx].get('color_hist')
                        )

                        # Check face similarity
                        face_dist = similarity_metrics.compute_hist_dist(
                            detections[det_idx].get('face_hist'),
                            detections[match_idx].get('face_hist')
                        )
                        
                        # Combined appearance distance
                        det_has_face = detections[det_idx].get('face_hist') is not None and np.sum(detections[det_idx].get('face_hist', np.array([]))) > 0
                        match_has_face = detections[match_idx].get('face_hist') is not None and np.sum(detections[match_idx].get('face_hist', np.array([]))) > 0
                        
                        if det_has_face and match_has_face:
                            hist_dist = 0.6 * face_dist + 0.4 * torso_dist
                        else:
                            hist_dist = torso_dist
                        
                        # FIXED: Require BOTH pose AND color similarity to mark as ghost
                        # This prevents marking a real person as ghost when they're close to another person
                        # Old logic: OR (too aggressive, caused real people to be suppressed)
                        # New logic: AND (both conditions must be true)
                        # Stricter thresholds: joint_dist < 0.3 AND hist_dist < 0.3
                        if joint_dist < 0.3 and hist_dist < 0.3:
                            is_ghost = True
                            break
                
                if not is_ghost:
                    final_unmatched_dets.append(det_idx)
                
        return matches, list(unmatched_tracks), final_unmatched_dets
    
    def update_tracks(self,
                      detections: List[Dict],
                      matches: Dict[int, int],
                      unmatched_tracks: List[int],
                      unmatched_dets: List[int],
                      frame: np.ndarray = None,
                      frame_idx: int = -1):
        """Update tracks with matched detections."""
        # Update matched
        for track_id, det_idx in matches.items():
            self.tracks[track_id].update(detections[det_idx])
        
        # Age unmatched tracks - check if they become "lost" for first time
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            was_active = track.time_since_update == 0
            track.time_since_update += 1
            
            # If track just became lost AND has embedding, keep it for gallery
            # (actual move to gallery happens when track is deleted)
        
        # Create new tracks
        new_track_ids = []
        for det_idx in unmatched_dets:
            new_track = MHRTrackState(track_id=self.next_id)
            new_track.initialize(detections[det_idx])
            new_track.creation_frame = frame_idx
            self.tracks[self.next_id] = new_track
            new_track_ids.append(self.next_id)
            self.next_id += 1
        
        # Try to extract face embeddings for new tracks
        if frame is not None and frame_idx >= 0:
            for track_id in new_track_ids:
                track = self.tracks[track_id]

                # Try to get face embedding immediately
                success, quality = self.face_embedding_manager.try_extract_embedding(track, frame, frame_idx)

                if success:

                    # Check if this face matches any lost track in gallery
                    old_tid = self.face_embedding_manager.check_gallery_for_match(track)
                    if old_tid is not None:
                        # Found a match! This is a returning person
                        self.track_merger.retrospective_merge(
                            old_tid, track_id, self.tracks,
                            self.face_embedding_manager.lost_track_gallery,
                            frame_idx, self.merge_events,
                            self.face_embedding_manager.remove_from_queue,
                            self.face_embedding_manager.remove_from_gallery
                        )
                    else:
                        # If no gallery match, check if face matches any OTHER active track
                        # This catches cases where the same face was incorrectly assigned to different bodies
                        active_match_id = self.face_embedding_manager.check_active_tracks_for_match(track, self.tracks)
                        if active_match_id is not None:
                            # Same face detected on two different tracks - merge them
                            # Merge into the lower ID (older track)
                            if active_match_id < track_id:
                                self.track_merger.merge_active_tracks(
                                    active_match_id, track_id, self.tracks, frame_idx, self.merge_events,
                                    self.face_embedding_manager.remove_from_queue
                                )
                            else:
                                self.track_merger.merge_active_tracks(
                                    track_id, active_match_id, self.tracks, frame_idx, self.merge_events,
                                    self.face_embedding_manager.remove_from_queue
                                )
                else:
                    # Schedule for deferred embedding extraction
                    self.face_embedding_manager.schedule_deferred_embedding(track_id, frame_idx)
        
        # Remove dead tracks - move their embeddings to gallery first
        to_remove = [tid for tid, t in self.tracks.items()
                     if t.time_since_update > self.max_age]
        for tid in to_remove:
            track = self.tracks[tid]
            self.face_embedding_manager.move_to_gallery(track, frame_idx)
            del self.tracks[tid]

            # Also remove from deferred queue and active search queue if present
            self.face_embedding_manager.remove_from_queue(tid)
            self.face_embedding_manager.remove_from_search_queue(tid)
    
    
    def process_frame(self,
                      frame: np.ndarray,
                      frame_idx: int,
                      temp_dir: str = "/tmp") -> Dict:
        """Process single frame."""
        # Camera tracking
        current_bboxes = [t.get_bbox() for t in self.tracks.values()]
        camera_motion = self.camera_tracker.track(frame, person_bboxes=current_bboxes)
        
        # Clear merge events for this frame
        self.merge_events = []
        
        # Run SAM 3D Body
        temp_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(temp_path, frame)
        
        try:
            detections = self.estimator.process_one_image(temp_path, bbox_thr=self.bbox_threshold, use_fp16=self.use_fp16)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Compute color and face histograms for detections
        for det in detections:
            det['color_hist'] = similarity_metrics.compute_color_hist(frame, det['bbox'])
            det['face_hist'] = similarity_metrics.compute_face_hist(frame, det['pred_joint_coords'], det['bbox'])
        
        # Match and update
        matches, unmatched_tracks, unmatched_dets = \
            self.match_detections_to_tracks(detections, camera_motion)
        self.update_tracks(detections, matches, unmatched_tracks, unmatched_dets, 
                          frame=frame, frame_idx=frame_idx)
        
        # 3D Ghost Merging: merge new tracks that are ghosts of lost tracks
        self.track_merger.merge_ghost_tracks(self.tracks)

        # Check deferred embeddings (for tracks that didn't have face on creation)
        def handle_face_match(old_tid, new_tid, is_active_match):
            """Callback for handling face matches from deferred embeddings."""
            if is_active_match:
                # Active track match - merge into lower ID
                if old_tid < new_tid:
                    self.track_merger.merge_active_tracks(
                        old_tid, new_tid, self.tracks, frame_idx, self.merge_events,
                        self.face_embedding_manager.remove_from_queue
                    )
                else:
                    self.track_merger.merge_active_tracks(
                        new_tid, old_tid, self.tracks, frame_idx, self.merge_events,
                        self.face_embedding_manager.remove_from_queue
                    )
            else:
                # Gallery match - retrospective merge
                self.track_merger.retrospective_merge(
                    old_tid, new_tid, self.tracks,
                    self.face_embedding_manager.lost_track_gallery,
                    frame_idx, self.merge_events,
                    self.face_embedding_manager.remove_from_queue,
                    self.face_embedding_manager.remove_from_gallery
                )

        self.face_embedding_manager.check_deferred_embeddings(
            frame, frame_idx, self.tracks, handle_face_match
        )

        # Check active face searches (for tracks with low-quality single faces)
        self.face_embedding_manager.check_active_face_searches(
            frame, frame_idx, self.tracks
        )

        # Check all tracks and schedule active face search if needed
        for track_id, track in self.tracks.items():
            self.face_embedding_manager.check_and_schedule_active_face_search(track, frame_idx)

        # Cleanup old gallery entries
        self.face_embedding_manager.cleanup_gallery(frame_idx)
        
        # Collect results
        results = {
            'frame_idx': frame_idx,
            'camera_motion': camera_motion,
            'num_detections': len(detections),
            'num_tracks': len(self.tracks),
            'persons': {},
            'merge_events': self.merge_events
        }
        
        for track_id, track in self.tracks.items():
            # Check distance from camera
            cam_t = track.get_last_cam_t()
            dist = 0.0
            if cam_t is not None:
                dist = np.linalg.norm(cam_t)
            
            # Skip if too far
            if dist > self.distance_limit:
                continue

            if track.hits >= self.min_hits:
                smoothed_joints = track.get_smoothed_joints()
                body_pose, hand_pose = track.get_smoothed_poses()
                
                # Helper to safely convert to list
                def safe_tolist(arr):
                    if arr is None:
                        return None
                    if isinstance(arr, np.ndarray):
                        return arr.tolist()
                    return list(arr)

                results['persons'][track_id] = {
                    'bbox': track.get_bbox(),
                    'joints': safe_tolist(smoothed_joints),
                    'body_pose': safe_tolist(body_pose),
                    'hand_pose': safe_tolist(hand_pose),
                    'shape': safe_tolist(track.shape_params),
                    'lhand_bbox': safe_tolist(track.lhand_bbox),
                    'rhand_bbox': safe_tolist(track.rhand_bbox),
                    # Add vertices and camera translation for 3D visualization
                    'vertices': safe_tolist(track.last_detection.get('pred_vertices')) if track.last_detection else None,
                    'cam_t': safe_tolist(track.last_detection.get('pred_cam_t')) if track.last_detection else None,
                    'face_crops': [
                        {
                            'image': emb['face_crop_b64'],
                            'quality': emb['quality'],
                            'frame': emb['frame']
                        }
                        for emb in track.face_embeddings
                    ] if track.face_embeddings else []
                }
        
        return results
