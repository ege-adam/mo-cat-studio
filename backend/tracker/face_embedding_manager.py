import numpy as np
from typing import Dict, Optional, Tuple


class FaceEmbeddingManager:
    """
    Manages face embeddings for track re-identification.

    Handles:
    - Face embedding extraction with retry logic
    - Lost track gallery for re-identification
    - Gallery cleanup
    - Face matching against gallery and active tracks
    """

    def __init__(self,
                 face_recognizer,
                 embed_retry_interval: int = 10,
                 embed_max_retries: int = 3,
                 embed_quality_threshold: float = 0.4,
                 gallery_max_age: int = 300,
                 face_match_threshold: float = 0.5,
                 max_embeddings_per_person: int = 10,
                 embed_similarity_threshold: float = 0.9,
                 distance_limit: float = 20.0):
        """
        Initialize FaceEmbeddingManager.

        Args:
            face_recognizer: Face recognition model (lazy-loaded property)
            embed_retry_interval: Frames between retry attempts
            embed_max_retries: Max retry attempts
            embed_quality_threshold: Min quality to accept embedding
            gallery_max_age: Frames to keep lost tracks in gallery
            face_match_threshold: Cosine similarity threshold for face match
            max_embeddings_per_person: Max embeddings to keep per track
            embed_similarity_threshold: Threshold for considering embeddings similar
            distance_limit: Max distance from camera to process faces (meters)
        """
        self.face_recognizer = face_recognizer

        # Lost track gallery for re-identification
        # { old_track_id: { 'embeddings': List[Dict], 'lost_frame': int,
        #                   'face_hist': np.ndarray, 'color_hist': np.ndarray } }
        self.lost_track_gallery: Dict[int, Dict] = {}

        # Deferred embedding queue for tracks that need face re-extraction
        # { track_id: { 'next_check_frame': int, 'attempts_remaining': int } }
        self.deferred_embed_queue: Dict[int, Dict] = {}

        # Active face search queue for tracks with low-quality single faces
        # { track_id: { 'next_search_frame': int, 'best_quality': float } }
        self.active_face_search_queue: Dict[int, Dict] = {}

        # Parameters
        self.embed_retry_interval = embed_retry_interval
        self.embed_max_retries = embed_max_retries
        self.embed_quality_threshold = embed_quality_threshold
        self.gallery_max_age = gallery_max_age
        self.face_match_threshold = face_match_threshold
        self.max_embeddings_per_person = max_embeddings_per_person
        self.embed_similarity_threshold = embed_similarity_threshold
        self.distance_limit = distance_limit

    def _try_extract_with_threshold(self, track, frame: np.ndarray, frame_idx: int,
                                     min_quality: float) -> Tuple[bool, float]:
        """
        Try to extract face embedding with a custom quality threshold.

        Internal method used for active face search with lowered threshold.

        Args:
            track: The track to extract embedding for
            frame: Current BGR frame
            frame_idx: Current frame index
            min_quality: Minimum quality threshold to accept face

        Returns:
            Tuple of (success, quality)
        """
        if not self.face_recognizer.is_available:
            return False, 0.0

        # Check distance before processing face
        cam_t = track.get_last_cam_t()
        if cam_t is not None:
            dist = np.linalg.norm(cam_t)
            if dist > self.distance_limit:
                return False, 0.0

        bbox = track.get_bbox()
        face_joints = None
        if track.last_detection is not None:
            joints = track.last_detection.get('pred_joint_coords')
            if joints is not None and len(joints) >= 5:
                face_joints = joints[:5]

        embedding, quality, face_crop_b64 = self.face_recognizer.get_embedding_for_person(
            frame, bbox, face_joints
        )

        track.embed_attempts += 1
        track.last_embed_frame = frame_idx

        if embedding is not None and quality >= min_quality:
            # Check if sufficiently different from existing embeddings
            should_add = True
            if track.face_embeddings:
                max_similarity = max(
                    self.face_recognizer.compute_similarity(embedding, old['embedding'])
                    for old in track.face_embeddings
                )
                # Only add if < embed_similarity_threshold (default 0.9)
                should_add = max_similarity < self.embed_similarity_threshold

            if should_add:
                track.face_embeddings.append({
                    'embedding': embedding,
                    'quality': quality,
                    'frame': frame_idx,
                    'face_crop_b64': face_crop_b64,
                    'bbox': bbox
                })

                # Keep top max_embeddings_per_person by quality
                track.face_embeddings.sort(key=lambda x: x['quality'], reverse=True)
                track.face_embeddings = track.face_embeddings[:self.max_embeddings_per_person]

                return True, quality

        return False, quality

    def try_extract_embedding(self, track, frame: np.ndarray, frame_idx: int) -> Tuple[bool, float]:
        """
        Try to extract face embedding for a track.

        Uses the default embed_quality_threshold (0.4 = 40%).

        Args:
            track: The track to extract embedding for
            frame: Current BGR frame
            frame_idx: Current frame index

        Returns:
            Tuple of (success, quality)
        """
        return self._try_extract_with_threshold(track, frame, frame_idx, self.embed_quality_threshold)

    def check_gallery_for_match(self, track) -> Optional[int]:
        """
        Check if a track's embedding matches any lost track in the gallery.

        Args:
            track: The track to check

        Returns:
            Matched old track ID if found, None otherwise
        """
        if not track.face_embeddings:
            return None

        best_match_id = None
        best_similarity = 0.0

        for old_tid, gallery_entry in self.lost_track_gallery.items():
            gallery_embeddings = gallery_entry.get('embeddings', [])
            if not gallery_embeddings:
                continue

            # Compare ALL track embeddings against ALL gallery embeddings
            # Use BEST match (max similarity) across all combinations
            for track_emb_dict in track.face_embeddings:
                for gallery_emb_dict in gallery_embeddings:
                    similarity = self.face_recognizer.compute_similarity(
                        track_emb_dict['embedding'],
                        gallery_emb_dict['embedding']
                    )

                    if similarity >= self.face_match_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = old_tid

        return best_match_id

    def check_active_tracks_for_match(self, track, tracks: Dict) -> Optional[int]:
        """
        Check if a track's embedding matches any OTHER active track.

        Args:
            track: The track to check
            tracks: Dictionary of all active tracks

        Returns:
            Matched active track ID if found, None otherwise
        """
        if not track.face_embeddings:
            return None

        best_match_id = None
        best_similarity = 0.0

        for other_id, other_track in tracks.items():
            if other_id == track.track_id:
                continue

            if not other_track.face_embeddings:
                continue

            # Compare ALL track embeddings against ALL other track embeddings
            for track_emb_dict in track.face_embeddings:
                for other_emb_dict in other_track.face_embeddings:
                    similarity = self.face_recognizer.compute_similarity(
                        track_emb_dict['embedding'],
                        other_emb_dict['embedding']
                    )

                    if similarity >= self.face_match_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = other_id

        return best_match_id

    def move_to_gallery(self, track, frame_idx: int):
        """
        Move a track's face embedding to the lost track gallery.

        Called when a track is about to be deleted.

        Args:
            track: The track to move to gallery
            frame_idx: Current frame index
        """
        if not track.face_embeddings:
            return

        self.lost_track_gallery[track.track_id] = {
            'embeddings': [emb.copy() for emb in track.face_embeddings],
            'lost_frame': frame_idx,
            'face_hist': track.face_hist.copy() if track.face_hist is not None else None,
            'color_hist': track.color_hist.copy() if track.color_hist is not None else None,
        }



    def cleanup_gallery(self, current_frame: int):
        """
        Remove old entries from the lost track gallery.

        Args:
            current_frame: Current frame index
        """
        to_remove = []
        for tid, entry in self.lost_track_gallery.items():
            if current_frame - entry['lost_frame'] > self.gallery_max_age:
                to_remove.append(tid)

        for tid in to_remove:
            del self.lost_track_gallery[tid]

    def schedule_deferred_embedding(self, track_id: int, current_frame: int):
        """
        Schedule a track for deferred face embedding extraction.

        Args:
            track_id: Track ID to schedule
            current_frame: Current frame index
        """
        self.deferred_embed_queue[track_id] = {
            'next_check_frame': current_frame + self.embed_retry_interval,
            'attempts_remaining': self.embed_max_retries
        }

    def check_deferred_embeddings(self, frame: np.ndarray, frame_idx: int,
                                   tracks: Dict, on_match_callback):
        """
        Check and process deferred embedding extractions.

        Called every frame to check if any tracks need face re-extraction.

        Args:
            frame: Current BGR frame
            frame_idx: Current frame index
            tracks: Dictionary of all active tracks
            on_match_callback: Callback function for handling matches
                              Signature: (old_tid, new_tid, is_active_match) -> None
        """
        tracks_to_remove = []

        for track_id, schedule in list(self.deferred_embed_queue.items()):
            if frame_idx < schedule['next_check_frame']:
                continue

            if track_id not in tracks:
                tracks_to_remove.append(track_id)
                continue

            track = tracks[track_id]

            # Try to extract embedding
            success, quality = self.try_extract_embedding(track, frame, frame_idx)

            if success:

                # Check gallery for retrospective match
                old_tid = self.check_gallery_for_match(track)
                if old_tid is not None:
                    on_match_callback(old_tid, track_id, False)  # Gallery match
                else:
                    # If no gallery match, check other ACTIVE tracks
                    active_match_id = self.check_active_tracks_for_match(track, tracks)
                    if active_match_id is not None:
                        on_match_callback(active_match_id, track_id, True)  # Active match

                tracks_to_remove.append(track_id)
            else:
                # Schedule next retry or give up
                schedule['attempts_remaining'] -= 1
                if schedule['attempts_remaining'] <= 0:
                    tracks_to_remove.append(track_id)
                else:
                    schedule['next_check_frame'] = frame_idx + self.embed_retry_interval

        for track_id in tracks_to_remove:
            if track_id in self.deferred_embed_queue:
                del self.deferred_embed_queue[track_id]

    def remove_from_queue(self, track_id: int):
        """
        Remove a track from the deferred embedding queue.

        Args:
            track_id: Track ID to remove
        """
        if track_id in self.deferred_embed_queue:
            del self.deferred_embed_queue[track_id]

    def remove_from_gallery(self, track_id: int):
        """
        Remove a track from the lost track gallery.

        Args:
            track_id: Track ID to remove
        """
        if track_id in self.lost_track_gallery:
            del self.lost_track_gallery[track_id]

    def check_and_schedule_active_face_search(self, track, frame_idx: int):
        """
        Check if a track needs active face search (single face with quality < 30%).
        If so, schedule it for periodic face quality improvement searches.

        Args:
            track: The track to check
            frame_idx: Current frame index
        """
        # Only consider tracks with exactly 1 face embedding
        if len(track.face_embeddings) != 1:
            # Remove from queue if no longer a single-face track
            if track.track_id in self.active_face_search_queue:
                del self.active_face_search_queue[track.track_id]
            return

        best_quality = track.face_embeddings[0]['quality']

        # If quality < 0.3 (30%), schedule for active search
        if best_quality < 0.3:
            if track.track_id not in self.active_face_search_queue:
                # First time scheduling - search on next interval * 2 frames
                search_interval = self.embed_retry_interval * 2
                self.active_face_search_queue[track.track_id] = {
                    'next_search_frame': frame_idx + search_interval,
                    'best_quality': best_quality
                }
        else:
            # Quality is good enough, remove from queue if present
            if track.track_id in self.active_face_search_queue:
                del self.active_face_search_queue[track.track_id]

    def check_active_face_searches(self, frame: np.ndarray, frame_idx: int, tracks: Dict):
        """
        Check and process active face search queue for quality improvement.

        Called every frame to check if any tracks need better face extraction.

        Args:
            frame: Current BGR frame
            frame_idx: Current frame index
            tracks: Dictionary of all active tracks
        """
        tracks_to_remove = []

        for track_id, search_info in list(self.active_face_search_queue.items()):
            if frame_idx < search_info['next_search_frame']:
                continue

            if track_id not in tracks:
                tracks_to_remove.append(track_id)
                continue

            track = tracks[track_id]

            # Verify still has single low-quality face
            if len(track.face_embeddings) != 1:
                tracks_to_remove.append(track_id)
                continue

            current_quality = track.face_embeddings[0]['quality']
            if current_quality >= 0.3:
                tracks_to_remove.append(track_id)
                continue

            # Try to extract better embedding with lowered threshold (0.3 = 30%)
            success, new_quality = self._try_extract_with_threshold(
                track, frame, frame_idx, min_quality=0.3
            )

            if success and new_quality >= 0.3:
                # The better face was already added as a variant
                # Stop searching since we reached 30% threshold
                tracks_to_remove.append(track_id)
            else:
                # No improvement, schedule next check
                search_interval = self.embed_retry_interval * 2
                search_info['next_search_frame'] = frame_idx + search_interval

        for track_id in tracks_to_remove:
            if track_id in self.active_face_search_queue:
                del self.active_face_search_queue[track_id]

    def remove_from_search_queue(self, track_id: int):
        """
        Remove a track from the active face search queue.

        Args:
            track_id: Track ID to remove
        """
        if track_id in self.active_face_search_queue:
            del self.active_face_search_queue[track_id]
