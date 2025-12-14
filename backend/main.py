from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import json
import asyncio
from typing import Dict
import queue
import traceback
from tracker_engine import TrackerEngine

# Global Tracker Engine (Singleton)
tracker_engine = None

# Last frame data for state recovery on frontend reconnect
last_frame_data = None
current_video_path = None
faces_data = []

# Track ranges history - stores all track segments for state recovery
# Format: { track_id: { "id": str, "segments": [{"start": int, "end": int}, ...], "color": str } }
all_track_ranges = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan for startup/shutdown events."""
    global tracker_engine
    print("=== LIFESPAN STARTUP ===")
    try:
        print("Initializing TrackerEngine...")
        tracker_engine = TrackerEngine()
        print("✓ Tracker Engine Initialized Successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Tracker Engine: {e}")
        traceback.print_exc()
        raise  # Don't start the app if initialization fails
    
    yield  # App runs here
    
    print("=== LIFESPAN SHUTDOWN ===")
    if tracker_engine:
        tracker_engine.stop()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.get("/state")
def get_state():
    """
    Get current tracker state for frontend recovery on page refresh.
    Returns the last processed frame data, video path, and track information with segments.
    """
    global last_frame_data, current_video_path, faces_data, all_track_ranges
    
    if not tracker_engine:
        return {"active": False, "initialized": False}
    
    return {
        "active": tracker_engine.is_processing if tracker_engine else False,
        "initialized": True,
        "video_path": current_video_path,
        "current_frame": last_frame_data.get("frame_idx") if last_frame_data else None,
        "persons": last_frame_data.get("persons") if last_frame_data else {},
        "image_base64": last_frame_data.get("image_base64") if last_frame_data else None,
        "faces": faces_data,
        "track_ranges": all_track_ranges
    }

# Gap threshold for creating new segments (in frames)
GAP_THRESHOLD = 5

# Track history indices - maps track_id to list of (segment_start_idx, segment_end_idx) 
# into the track's mhr_params_history array
track_history_indices = {}

def update_track_ranges(frame_idx: int, person_ids: list, tracker):
    """
    Update all_track_ranges with current frame's persons.
    Creates new segments when a person reappears after a gap.
    Also stores indices into the mhr_params_history for segment export.
    """
    global all_track_ranges, track_history_indices
    
    for person_id in person_ids:
        track_id = str(person_id)
        int_track_id = int(person_id)
        
        # Get current history length for this track
        history_len = 0
        face_crops = []
        if tracker and tracker.tracks.get(int_track_id):
            track_obj = tracker.tracks[int_track_id]
            if track_obj.mhr_params_history:
                history_len = len(track_obj.mhr_params_history)
            
            # Extract face crops
            if track_obj.face_embeddings:
                face_crops = [
                    {
                        'image': emb['face_crop_b64'],
                        'quality': emb['quality'],
                        'frame': emb['frame']
                    }
                    for emb in track_obj.face_embeddings
                ]
        
        if track_id not in all_track_ranges:
            # Candy color palette matching frontend
            CANDY_COLORS = [
                '#FF6B9D',  # Pink
                '#C96EF5',  # Purple
                '#6EC3F5',  # Blue
                '#5EEEFF',  # Cyan
                '#6EFFC6',  # Mint
                '#FFE66D',  # Yellow
                '#FFAA6D',  # Orange
                '#FF8B7B',  # Coral
            ]
            color_index = int(person_id) % len(CANDY_COLORS)

            # New track - create with first segment
            all_track_ranges[track_id] = {
                "id": track_id,
                "segments": [{
                    "start": frame_idx,
                    "end": frame_idx,
                    "history_start": 0,
                    "history_end": history_len - 1 if history_len > 0 else 0
                }],
                "color": CANDY_COLORS[color_index],
                "_last_seen": frame_idx,
                "_last_history_idx": history_len - 1 if history_len > 0 else 0,
                "face_crops": face_crops
            }
        else:
            track = all_track_ranges[track_id]
            # Update face crops
            track["face_crops"] = face_crops
            gap = frame_idx - track.get("_last_seen", 0)
            
            if gap > GAP_THRESHOLD:
                # Person reappeared after gap - start new segment
                # New segment starts at current history index
                track["segments"].append({
                    "start": frame_idx, 
                    "end": frame_idx,
                    "history_start": history_len - 1 if history_len > 0 else 0,
                    "history_end": history_len - 1 if history_len > 0 else 0
                })
            else:
                # Extend current segment
                if track["segments"]:
                    track["segments"][-1]["end"] = frame_idx
                    track["segments"][-1]["history_end"] = history_len - 1 if history_len > 0 else 0
            
            track["_last_seen"] = frame_idx
            track["_last_history_idx"] = history_len - 1 if history_len > 0 else 0

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"Client {client_id} connected")
    
    # Queue for results from tracker thread
    result_queue = queue.Queue()
    
    # Task to poll queue and send to websocket
    async def sender_task():
        while True:
            try:
                # Poll queue
                while not result_queue.empty():
                    item = result_queue.get_nowait()
                    await websocket.send_json(item)
                await asyncio.sleep(0.01)
            except Exception as e:
                # Websocket closed or other error
                break

    sender = asyncio.create_task(sender_task())

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                # Check if tracker is initialized
                if tracker_engine is None:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Tracker engine not initialized. Check server logs for initialization errors."
                    })
                    continue
                
                video_path = message.get("video_path")
                start_frame = int(message.get("start_frame", 0))
                skip_frames = int(message.get("skip_frames", 1))
                distance_limit = float(message.get("distanceLimit", 20.0))
                ghosting_threshold = float(message.get("ghosting_threshold", 0.25))
                use_gpu = bool(message.get("use_gpu", True))
                max_age = int(message.get("max_age", 30))
                bbox_threshold = float(message.get("bbox_threshold", 0.5))
                resize_factor = float(message.get("resize_factor", 1.0))
                use_fp16 = bool(message.get("use_fp16", False))
                disable_ghost_suppression = bool(message.get("disable_ghost_suppression", False))
                ghost_3d_threshold = float(message.get("ghost_3d_threshold", 0.5))
                hist_match_threshold = float(message.get("hist_match_threshold", 0.3))
                
                # Face recognition parameters
                enable_face_recognition = bool(message.get("enable_face_recognition", True))
                embed_retry_interval = int(message.get("embed_retry_interval", 10))
                embed_max_retries = int(message.get("embed_max_retries", 3))
                embed_quality_threshold = float(message.get("embed_quality_threshold", 0.4))
                gallery_max_age = int(message.get("gallery_max_age", 300))
                face_match_threshold = float(message.get("face_match_threshold", 0.5))
                max_embeddings_per_person = int(message.get("maxEmbeddingsPerPerson", 10))
                embed_similarity_threshold = float(message.get("embedSimilarityThreshold", 0.9))
                
                if not video_path or not os.path.exists(video_path):
                    await websocket.send_json({"type": "error", "message": "Video file not found"})
                    continue
                
                # Get faces from estimator if available
                faces = []
                if tracker_engine and tracker_engine.estimator:
                    try:
                        # Flatten faces to 1D array for frontend
                        faces = tracker_engine.estimator.faces.flatten().tolist()
                    except Exception as e:
                        print(f"Error getting faces: {e}")

                await websocket.send_json({
                    "type": "status", 
                    "message": "Processing started",
                    "faces": faces
                })
                
                # Stop existing processing if any
                if tracker_engine:
                    tracker_engine.stop()

                # Run processing in a separate thread to avoid blocking event loop
                # But we need to yield results to websocket.
                # We can iterate over the generator.
                
                # Since generator blocks, we should run it in a thread executor?
                # Or just iterate? If we iterate, we block the event loop?
                # Yes, tracker.process_video is synchronous.
                # We need to run it in a thread and put results in a queue, or use run_in_executor.
                
                # Let's use a simple loop with asyncio.sleep(0) to yield control if possible, 
                # but the generator itself is blocking CPU bound.
                # Better: run the generator in a thread, and have it put results in an asyncio.Queue?
                # Or just iterate and accept that we block the loop slightly? 
                # No, blocking loop is bad for websocket pings.
                
                # Let's use run_in_executor for the whole process? No, we need streaming.
                
                # We can use a thread that pushes to a queue.
                
                def run_tracker():
                    global last_frame_data, current_video_path, faces_data, all_track_ranges
                    current_video_path = video_path
                    faces_data = faces
                    # Clear track ranges for new processing session
                    all_track_ranges = {}
                    try:
                        for frame_result in tracker_engine.process_video(
                            video_path, start_frame, skip_frames, ghosting_threshold, 
                            use_gpu, max_age, bbox_threshold, resize_factor, use_fp16, 
                            disable_ghost_suppression, ghost_3d_threshold, hist_match_threshold,
                            enable_face_recognition, embed_retry_interval, embed_max_retries,
                            embed_quality_threshold, gallery_max_age, face_match_threshold,
                            max_embeddings_per_person, embed_similarity_threshold, distance_limit
                        ):
                            # Store last frame for state recovery
                            last_frame_data = _serialize_frame(frame_result)
                            
                            # Update track ranges with segments
                            frame_idx = frame_result['frame_idx']
                            person_ids = list(frame_result['persons'].keys())
                            update_track_ranges(frame_idx, person_ids, tracker_engine.tracker)
                            
                            result_queue.put({"type": "frame", "data": last_frame_data})
                        result_queue.put({"type": "complete"})
                    except Exception as e:
                        print(f"Tracker error: {e}")
                        result_queue.put({"type": "error", "message": str(e)})

                import threading
                t = threading.Thread(target=run_tracker)
                t.start()
                
            elif message.get("type") == "stop":
                if tracker_engine:
                    tracker_engine.stop()
                await websocket.send_json({"type": "status", "message": "Stopping..."})

            elif message.get("type") == "pause":
                if tracker_engine:
                    tracker_engine.pause()
                    await websocket.send_json({"type": "paused", "message": "Processing paused"})
                else:
                    await websocket.send_json({"type": "error", "message": "Tracker not initialized"})

            elif message.get("type") == "resume":
                if tracker_engine:
                    tracker_engine.resume()
                    await websocket.send_json({"type": "resumed", "message": "Processing resumed"})
                else:
                    await websocket.send_json({"type": "error", "message": "Tracker not initialized"})

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        if tracker_engine:
            tracker_engine.stop()
    finally:
        sender.cancel()

def _serialize_frame(frame_result):
    """
    Convert numpy arrays to lists for JSON serialization.
    NOTE: tracker.py now handles most of this, so we just pass it through.
    We ensure image_base64 is included.
    """
    return {
        'frame_idx': frame_result['frame_idx'],
        'persons': frame_result['persons'],
        'image_base64': frame_result.get('image_base64'),
        'merge_events': frame_result.get('merge_events', [])
    }

from fastapi.responses import FileResponse
from tracker.exporter import MHRExporter

@app.get("/export/{track_id}/segment/{segment_idx}")
def export_segment(track_id: int, segment_idx: int):
    """
    Export a specific segment of a track to GLB format.
    
    Args:
        track_id: The ID of the track to export
        segment_idx: The index of the segment within the track
    """
    global all_track_ranges
    
    if not tracker_engine or not tracker_engine.tracker:
        return {"error": "Tracker not initialized"}
    
    track = tracker_engine.tracker.tracks.get(track_id)
    if not track:
        return {"error": "Track not found"}
    
    # Get segment info from all_track_ranges
    track_range = all_track_ranges.get(str(track_id))
    if not track_range:
        return {"error": "Track range not found"}
    
    segments = track_range.get("segments", [])
    if segment_idx < 0 or segment_idx >= len(segments):
        return {"error": f"Segment index {segment_idx} out of range (0-{len(segments)-1})"}
    
    segment = segments[segment_idx]
    history_start = segment.get("history_start", 0)
    history_end = segment.get("history_end", len(track.mhr_params_history) - 1)
    
    # Slice the history for this segment
    if not track.mhr_params_history or len(track.mhr_params_history) == 0:
        return {"error": "No MHR model parameters available for this track"}
    
    segment_mhr_params = track.mhr_params_history[history_start:history_end + 1]
    segment_cam_t = track.cam_t_history[history_start:history_end + 1] if track.cam_t_history else None
    
    if len(segment_mhr_params) == 0:
        return {"error": "No frames in this segment"}
    
    # Get MHR model
    mhr_model = None
    try:
        if tracker_engine.estimator and hasattr(tracker_engine.estimator, 'model'):
            if hasattr(tracker_engine.estimator.model, 'head_pose'):
                mhr_model = tracker_engine.estimator.model.head_pose.mhr
    except Exception as e:
        print(f"Could not get MHR model: {e}")
    
    exporter = MHRExporter(mhr_model=mhr_model)
    
    # Export
    filename = f"track_{track_id}_seg{segment_idx}.glb"
    temp_path = os.path.join(UPLOAD_DIR, f"track_{track_id}_seg{segment_idx}.fbx")
    success = exporter.export_to_fbx(segment_mhr_params, temp_path, cam_t_history=segment_cam_t)
    if not success:
        return {"error": "GLB export failed - check server logs for details"}
    output_path = os.path.join(UPLOAD_DIR, filename)
    media_type = 'model/gltf-binary'

    if not os.path.exists(output_path):
        return {"error": f"Export failed - output file not created at {output_path}"}
    
    return FileResponse(output_path, media_type=media_type, filename=filename)

@app.get("/export/{track_id}")
def export_track(track_id: int):
    """
    Export a track to GLB format.
    
    Args:
        track_id: The ID of the track to export
    """
    if not tracker_engine or not tracker_engine.tracker:
        return {"error": "Tracker not initialized"}
    
    track = tracker_engine.tracker.tracks.get(track_id)
    if not track:
        return {"error": "Track not found"}
    
    # Check for MHR model params (preferred for accurate export)
    if not track.mhr_params_history or len(track.mhr_params_history) == 0:
        return {"error": "No MHR model parameters available for this track"}
    
    # Get MHR model from estimator to access character for native export
    mhr_model = None
    try:
        if tracker_engine.estimator and hasattr(tracker_engine.estimator, 'model'):
            if hasattr(tracker_engine.estimator.model, 'head_pose'):
                mhr_model = tracker_engine.estimator.model.head_pose.mhr
    except Exception as e:
        print(f"Could not get MHR model: {e}")
    
    exporter = MHRExporter(mhr_model=mhr_model)
    
    # Default to GLB (exports as .glb even if .fbx requested for compatibility)
    filename = f"track_{track_id}.glb"
    # Start with .fbx path, exporter will convert to .glb
    temp_path = os.path.join(UPLOAD_DIR, f"track_{track_id}.fbx")
    success = exporter.export_to_fbx(track.mhr_params_history, temp_path, cam_t_history=track.cam_t_history)
    if not success:
        return {"error": "GLB export failed - check server logs for details"}
    # The actual file will be .glb
    output_path = os.path.join(UPLOAD_DIR, filename)
    media_type = 'model/gltf-binary'

    if not os.path.exists(output_path):
        return {"error": f"Export failed - output file not created at {output_path}"}
    
    return FileResponse(output_path, media_type=media_type, filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
