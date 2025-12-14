import numpy as np
from typing import Dict, List
from .track_state import MHRTrackState
from . import similarity_metrics


class TrackMerger:
    """
    Handles merging of tracks for ghost suppression and re-identification.

    Manages:
    - Active track merging (when two active tracks are the same person)
    - Retrospective merging (when a new track matches a lost track)
    - 3D ghost detection and merging
    """

    def __init__(self,
                 face_recognizer,
                 disable_ghost_suppression: bool = False,
                 ghost_3d_threshold: float = 0.5,
                 hist_match_threshold: float = 0.3,
                 max_embeddings_per_person: int = 10,
                 embed_similarity_threshold: float = 0.9,
                 min_hits: int = 3):
        """
        Initialize TrackMerger.

        Args:
            face_recognizer: Face recognition model
            disable_ghost_suppression: Whether to disable ghost suppression
            ghost_3d_threshold: 3D distance threshold in meters for ghost merging
            hist_match_threshold: Histogram similarity threshold for ghost merging
            max_embeddings_per_person: Max embeddings to keep per track
            embed_similarity_threshold: Threshold for considering embeddings similar
            min_hits: Minimum hits for a track to be considered established
        """
        self.face_recognizer = face_recognizer
        self.disable_ghost_suppression = disable_ghost_suppression
        self.ghost_3d_threshold = ghost_3d_threshold
        self.hist_match_threshold = hist_match_threshold
        self.max_embeddings_per_person = max_embeddings_per_person
        self.embed_similarity_threshold = embed_similarity_threshold
        self.min_hits = min_hits

    def merge_active_tracks(self, keep_id: int, merge_id: int, tracks: Dict[int, MHRTrackState],
                            current_frame: int, merge_events: List[Dict],
                            remove_from_queue_callback) -> None:
        """
        Merge two active tracks into one.

        Args:
            keep_id: The track ID to keep (usually older/lower ID)
            merge_id: The track ID to merge into keep_id and then delete
            tracks: Dictionary of all tracks
            current_frame: Current frame index
            merge_events: List to append merge event to
            remove_from_queue_callback: Callback to remove track from deferred queue
        """
        if keep_id not in tracks or merge_id not in tracks:
            return

        print(f"Merging active track {merge_id} into {keep_id}")

        keep_track = tracks[keep_id]
        merge_track = tracks[merge_id]

        # Merge face embeddings
        for emb in merge_track.face_embeddings:
            # Check if similar embedding already exists
            is_new = True
            for existing in keep_track.face_embeddings:
                sim = self.face_recognizer.compute_similarity(emb['embedding'], existing['embedding'])
                if sim > self.embed_similarity_threshold:
                    is_new = False
                    break

            if is_new:
                keep_track.face_embeddings.append(emb)

        # Sort and limit embeddings
        keep_track.face_embeddings.sort(key=lambda x: x['quality'], reverse=True)
        keep_track.face_embeddings = keep_track.face_embeddings[:self.max_embeddings_per_person]

        # Record merge event
        merge_events.append({
            'old_id': merge_id,
            'new_id': keep_id,
            'frame': current_frame
        })

        # Remove merged track
        del tracks[merge_id]

        # Remove from deferred queue
        remove_from_queue_callback(merge_id)

    def retrospective_merge(self, old_track_id: int, new_track_id: int, tracks: Dict[int, MHRTrackState],
                            lost_track_gallery: Dict, current_frame: int, merge_events: List[Dict],
                            remove_from_queue_callback, remove_from_gallery_callback) -> None:
        """
        Merge a new track into an old track when face match is found.

        This handles the case where a person left and returned, and we
        identified them via face embedding.

        Args:
            old_track_id: The original track ID from gallery
            new_track_id: The new track ID to merge
            tracks: Dictionary of all tracks
            lost_track_gallery: Lost track gallery dictionary
            current_frame: Current frame index
            merge_events: List to append merge event to
            remove_from_queue_callback: Callback to remove track from deferred queue
            remove_from_gallery_callback: Callback to remove track from gallery
        """
        if new_track_id not in tracks:
            print(f"Retrospective merge failed: new track {new_track_id} not found")
            return

        new_track = tracks[new_track_id]
        gallery_entry = lost_track_gallery.get(old_track_id)

        if gallery_entry is None:
            print(f"Retrospective merge failed: old track {old_track_id} not in gallery")
            return

        print(f"Retrospective merge: track {new_track_id} -> old track {old_track_id}")

        # Create a restored track with the old ID
        restored_track = MHRTrackState(track_id=old_track_id)

        # Copy all data from new track
        restored_track.bbox_kf = new_track.bbox_kf
        restored_track.joint_history = new_track.joint_history.copy()
        restored_track.body_pose_history = new_track.body_pose_history.copy()
        restored_track.hand_pose_history = new_track.hand_pose_history.copy()
        restored_track.full_joint_history = new_track.full_joint_history.copy()
        restored_track.full_rot_history = new_track.full_rot_history.copy()
        restored_track.mhr_params_history = new_track.mhr_params_history.copy()
        restored_track.cam_t_history = new_track.cam_t_history.copy()
        restored_track.shape_params = new_track.shape_params
        restored_track.bone_lengths = new_track.bone_lengths
        restored_track.lhand_bbox = new_track.lhand_bbox
        restored_track.rhand_bbox = new_track.rhand_bbox
        restored_track.last_detection = new_track.last_detection
        restored_track.color_hist = new_track.color_hist
        restored_track.face_hist = new_track.face_hist
        restored_track.face_hist = new_track.face_hist
        restored_track.face_embeddings = new_track.face_embeddings.copy()
        restored_track.last_embed_frame = new_track.last_embed_frame
        restored_track.embed_attempts = new_track.embed_attempts
        restored_track.age = new_track.age
        restored_track.hits = new_track.hits
        restored_track.time_since_update = new_track.time_since_update
        restored_track.creation_frame = gallery_entry.get('lost_frame', -1)

        # Remove new track and add restored track
        del tracks[new_track_id]
        tracks[old_track_id] = restored_track

        # Remove from gallery
        remove_from_gallery_callback(old_track_id)

        # Remove from deferred queue if present
        remove_from_queue_callback(new_track_id)

        print(f"Retrospective merge complete: restored track {old_track_id}")

        # Record merge event for frontend
        merge_events.append({
            'old_id': new_track_id,
            'new_id': old_track_id,
            'frame': current_frame
        })

    def merge_ghost_tracks(self, tracks: Dict[int, MHRTrackState]) -> None:
        """
        Merge new tracks that are actually ghosts of lost tracks.

        When a person disappears and a new detection appears nearby with similar appearance,
        merge them by transferring the new track's data to the lost track and deleting the new track.

        Uses 3D position (cam_t) and histogram similarity for matching.

        Args:
            tracks: Dictionary of all tracks
        """
        if self.disable_ghost_suppression:
            return

        # Find lost tracks (time_since_update > 0) and new tracks (hits < min_hits)
        lost_tracks = [(tid, t) for tid, t in tracks.items()
                       if t.time_since_update > 0 and t.hits >= self.min_hits]
        new_tracks = [(tid, t) for tid, t in tracks.items()
                      if t.hits < self.min_hits and t.time_since_update == 0]

        if not lost_tracks or not new_tracks:
            return

        tracks_to_delete = []

        for new_tid, new_track in new_tracks:
            # Get new track's 3D position
            new_cam_t = new_track.get_last_cam_t()
            if new_cam_t is None:
                continue

            best_lost_tid = None
            best_score = float('inf')

            for lost_tid, lost_track in lost_tracks:
                # Skip if already scheduled for merge
                if lost_tid in [x[1] for x in tracks_to_delete]:
                    continue

                # Get lost track's last known 3D position
                lost_cam_t = lost_track.get_last_cam_t()
                if lost_cam_t is None:
                    continue

                # Compute 3D distance
                dist_3d = np.linalg.norm(np.array(new_cam_t) - np.array(lost_cam_t))

                # Check if within 3D threshold
                if dist_3d > self.ghost_3d_threshold:
                    continue

                # Compute combined histogram distance (face + torso)
                torso_dist = similarity_metrics.compute_hist_dist(new_track.color_hist, lost_track.color_hist)
                face_dist = similarity_metrics.compute_hist_dist(new_track.face_hist, lost_track.face_hist)

                # Check if face histograms are available
                new_has_face = new_track.face_hist is not None and np.sum(new_track.face_hist) > 0
                lost_has_face = lost_track.face_hist is not None and np.sum(lost_track.face_hist) > 0

                if new_has_face and lost_has_face:
                    hist_dist = 0.6 * face_dist + 0.4 * torso_dist
                else:
                    hist_dist = torso_dist

                # Check if histogram is similar enough
                if hist_dist > self.hist_match_threshold:
                    continue

                # Combined score (lower is better)
                score = dist_3d + hist_dist

                if score < best_score:
                    best_score = score
                    best_lost_tid = lost_tid

            if best_lost_tid is not None:
                # Found a match! Merge new track into lost track
                lost_track = tracks[best_lost_tid]

                # Compute final distance for logging
                torso_d = similarity_metrics.compute_hist_dist(new_track.color_hist, lost_track.color_hist)
                face_d = similarity_metrics.compute_hist_dist(new_track.face_hist, lost_track.face_hist)

                print(f"3D Ghost Merge: new track {new_tid} -> lost track {best_lost_tid} "
                      f"(3D dist: {np.linalg.norm(np.array(new_cam_t) - np.array(lost_track.get_last_cam_t())):.3f}m, "
                      f"torso_dist: {torso_d:.3f}, face_dist: {face_d:.3f})")

                # Transfer the new track's last detection to the lost track
                if new_track.last_detection:
                    lost_track.update(new_track.last_detection)

                # Schedule new track for deletion
                tracks_to_delete.append((new_tid, best_lost_tid))

        # Delete merged new tracks
        for new_tid, lost_tid in tracks_to_delete:
            if new_tid in tracks:
                del tracks[new_tid]
