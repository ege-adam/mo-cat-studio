import sys
import os
import cv2
import numpy as np
import threading
import queue
import time
from typing import Generator, Dict, Any

# Add sam-3d-body to path (now local)
# sys.path.append(os.path.join(os.path.dirname(__file__), "sam_3d_body"))

try:
    from tracker.tracker import MHRVideoTracker
    from model_factory import setup_sam_3d_body
except ImportError as e:
    print(f"Error importing tracker: {e}")
    raise

class TrackerEngine:
    def __init__(self, model_id='facebook/sam-3d-body-vith', use_gpu=True):
        print(f"Initializing TrackerEngine with model: {model_id}")
        self.estimator = setup_sam_3d_body(hf_repo_id=model_id)
        self.use_gpu = use_gpu
        self.tracker = MHRVideoTracker(self.estimator, use_gpu=use_gpu)
        self.is_processing = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start in unpaused state (event is set = not paused)

    def process_video(self, video_path: str, start_frame: int = 0, skip_frames: int = 1, ghosting_threshold: float = 0.25, use_gpu: bool = True, max_age: int = 30, bbox_threshold: float = 0.5, resize_factor: float = 1.0, use_fp16: bool = False, disable_ghost_suppression: bool = False, ghost_3d_threshold: float = 0.5, hist_match_threshold: float = 0.3, enable_face_recognition: bool = True, embed_retry_interval: int = 10, embed_max_retries: int = 3, embed_quality_threshold: float = 0.4, gallery_max_age: int = 300, face_match_threshold: float = 0.5, max_embeddings_per_person: int = 10, embed_similarity_threshold: float = 0.9, distance_limit: float = 20.0) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields tracking results for each processed frame.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path} ({total_frames} frames)")

        self.is_processing = True
        self.stop_event.clear()
        
        # Initialize tracker imports
        from tracker.tracker import MHRVideoTracker, CameraTracker
        
        # Re-initialize tracker for new video, passing parameters
        self.tracker = MHRVideoTracker(
            self.estimator,
            use_gpu=use_gpu,
            ghosting_threshold=ghosting_threshold,
            max_age=max_age,
            bbox_threshold=bbox_threshold,
            use_fp16=use_fp16,
            disable_ghost_suppression=disable_ghost_suppression,
            ghost_3d_threshold=ghost_3d_threshold,
            hist_match_threshold=hist_match_threshold,
            max_embeddings_per_person=max_embeddings_per_person,
            embed_similarity_threshold=embed_similarity_threshold,
            distance_limit=distance_limit,
            embed_retry_interval=embed_retry_interval,
            embed_max_retries=embed_max_retries,
            embed_quality_threshold=embed_quality_threshold,
            gallery_max_age=gallery_max_age,
            face_match_threshold=face_match_threshold
        )

        # Disable face recognition if requested
        if not enable_face_recognition:
            # Disable face recognition by setting quality threshold impossibly high
            self.tracker.face_embedding_manager.embed_quality_threshold = 999.0

        self.camera_tracker = CameraTracker()
        self.tracker.camera_tracker = self.camera_tracker # Assign the new camera tracker
        self.tracker.camera_tracker_idx = 0 # Reset camera tracker index

        frame_idx = 0
        processed_frames = 0
        try:
            while cap.isOpened():
                if self.stop_event.is_set():
                    print("Processing stopped by user.")
                    break

                # Check for pause - wait until unpaused
                self.pause_event.wait()

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < start_frame:
                    frame_idx += 1
                    continue

                if (frame_idx - start_frame) % skip_frames == 0:
                    # Resize frame if needed
                    if resize_factor != 1.0:
                        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

                    # Process frame using MHRVideoTracker logic
                    # We replicate process_frame logic here to avoid writing to disk if possible,
                    # but MHRVideoTracker.process_frame writes to temp file.
                    # We can use the existing process_frame method.

                    # Note: process_frame writes to /tmp/frame_X.jpg.
                    # We should ensure thread safety if multiple requests (but we likely limit to 1 job).

                    # Print progress
                    processed_frames += 1
                    print(f"Processing frame {frame_idx}/{total_frames} ({processed_frames} frames processed)")

                    results = self.tracker.process_frame(frame, frame_idx)

                    # Encode frame to base64 for streaming
                    _, buffer = cv2.imencode('.jpg', frame)
                    import base64
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    results['image_base64'] = img_str

                    # Yield result
                    yield results

                frame_idx += 1
        finally:
            cap.release()
            self.is_processing = False
            print("Processing finished.")

    def stop(self):
        """Stop current processing."""
        self.stop_event.set()

    def pause(self):
        """Pause current processing."""
        self.pause_event.clear()
        print("Processing paused.")

    def resume(self):
        """Resume current processing."""
        self.pause_event.set()
        print("Processing resumed.")

    def is_paused(self):
        """Check if processing is paused."""
        return not self.pause_event.is_set()
