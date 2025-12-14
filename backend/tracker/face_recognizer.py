"""
Face Recognition module using InsightFace for face detection and embedding extraction.

This module provides:
- Face detection using RetinaFace
- Face embedding extraction using ArcFace
- Quality scoring for face detections
- Similarity comparison between embeddings
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import cv2
import base64


class FaceRecognizer:
    """
    Face recognition wrapper using InsightFace.
    
    Provides face detection and 512-dimensional embedding extraction
    for person re-identification.
    
    Lazy-loaded to avoid startup overhead when not needed.
    """
    
    def __init__(self, 
                 model_name: str = 'buffalo_l',
                 device: str = 'cuda',
                 det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize FaceRecognizer.
        
        Args:
            model_name: InsightFace model pack name.
                - 'buffalo_l': Large model, best accuracy (~500MB)
                - 'buffalo_s': Small model, faster (~100MB)
            device: 'cuda' or 'cpu'
            det_size: Detection input size (width, height)
        """
        self.model_name = model_name
        self.device = device
        self.det_size = det_size
        self._app = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization of InsightFace models."""
        if self._initialized:
            return
            
        try:
            from insightface.app import FaceAnalysis
            
            providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            
            self._app = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            self._app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=self.det_size)
            self._initialized = True
            print(f"FaceRecognizer initialized with model={self.model_name}, device={self.device}")
            
        except ImportError as e:
            print(f"Warning: InsightFace not available. Face embedding will be disabled. Error: {e}")
            self._app = None
            self._initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize InsightFace: {e}")
            self._app = None
            self._initialized = True
    
    @property
    def is_available(self) -> bool:
        """Check if face recognition is available."""
        self._lazy_init()
        return self._app is not None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in an image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            List of face dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'det_score': detection confidence (0-1)
                - 'embedding': 512-dim vector
                - 'landmark_2d_106': 106 facial landmarks (if available)
        """
        self._lazy_init()
        
        if self._app is None:
            return []
        
        try:
            faces = self._app.get(image)
            
            results = []
            for face in faces:
                result = {
                    'bbox': face.bbox.tolist(),
                    'det_score': float(face.det_score),
                    'embedding': face.embedding if hasattr(face, 'embedding') else None,
                }
                if hasattr(face, 'landmark_2d_106'):
                    result['landmark_2d_106'] = face.landmark_2d_106
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def get_embedding_for_person(self, 
                                  image: np.ndarray, 
                                  person_bbox: List[float],
                                  face_joints: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """
        Get face embedding for a specific person given their body bbox.
        
        This method:
        1. Crops the person region (upper 50% for face area)
        2. Runs face detection
        3. Matches detected face to the person bbox
        4. Returns embedding and quality score
        
        Args:
            image: Full BGR image
            person_bbox: Person bounding box [x1, y1, x2, y2]
            face_joints: Optional face keypoints from MHR (indices 0-4)
            
        Returns:
            Tuple of (embedding, quality, face_crop_b64):
                - embedding: 512-dim vector or None if no face detected
                - quality: 0.0-1.0 confidence score
                - face_crop_b64: Base64 encoded JPEG string of the face crop
        """
        self._lazy_init()
        
        if self._app is None:
            return None, 0.0, None
        
        # Expand person bbox slightly and focus on upper portion
        x1, y1, x2, y2 = map(int, person_bbox)
        h, w = image.shape[:2]
        
        # Focus on upper 60% of body (head region)
        body_height = y2 - y1
        upper_y2 = y1 + int(body_height * 0.6)
        
        # Expand horizontally by 20% to catch faces at edges
        body_width = x2 - x1
        expand_x = int(body_width * 0.2)
        
        crop_x1 = max(0, x1 - expand_x)
        crop_y1 = max(0, y1)
        crop_x2 = min(w, x2 + expand_x)
        crop_y2 = min(h, upper_y2)
        
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, 0.0, None
        
        # Crop and detect faces
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        faces = self.detect_faces(crop)
        
        if not faces:
            return None, 0.0, None
        
        # If we have face keypoints from MHR, use them to select the best face
        best_face = None
        best_score = 0.0
        
        if face_joints is not None and len(face_joints) >= 5:
            # Use nose position (index 0) as reference
            nose_x, nose_y = face_joints[0, 0], face_joints[0, 1]
            
            for face in faces:
                face_bbox = face['bbox']
                # Convert crop-relative bbox to image-relative
                face_x1 = face_bbox[0] + crop_x1
                face_y1 = face_bbox[1] + crop_y1
                face_x2 = face_bbox[2] + crop_x1
                face_y2 = face_bbox[3] + crop_y1
                
                # Check if nose is inside face bbox
                if face_x1 <= nose_x <= face_x2 and face_y1 <= nose_y <= face_y2:
                    if face['det_score'] > best_score:
                        best_score = face['det_score']
                        best_face = face
        
        # Fallback: use highest confidence face
        if best_face is None:
            best_face = max(faces, key=lambda f: f['det_score'])
            best_score = best_face['det_score']
        
        embedding = best_face.get('embedding')
        if embedding is None:
            return None, 0.0, None
        
        # Quality is based on detection score and face size
        face_bbox = best_face['bbox']
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        face_area = face_width * face_height
        
        # Normalize quality: higher for larger, high-confidence faces
        # Min acceptable face: 48x48 pixels, optimal: 112x112+
        size_quality = min(1.0, face_area / (112 * 112))
        quality = best_score * size_quality
        
        # Extract face crop
        face_x1 = int(max(0, best_face['bbox'][0]))
        face_y1 = int(max(0, best_face['bbox'][1]))
        face_x2 = int(min(crop.shape[1], best_face['bbox'][2]))
        face_y2 = int(min(crop.shape[0], best_face['bbox'][3]))
        
        face_crop = crop[face_y1:face_y2, face_x1:face_x2]
        
        # Encode to base64 JPEG
        face_crop_b64 = None
        if face_crop.size > 0:
            try:
                _, buffer = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                face_crop_b64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Error encoding face crop: {e}")
        
        return embedding, quality, face_crop_b64
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First 512-dim embedding
            emb2: Second 512-dim embedding
            
        Returns:
            Similarity score in [-1, 1], typically [0.3, 1.0] for faces
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def is_same_person(self, 
                       emb1: np.ndarray, 
                       emb2: np.ndarray, 
                       threshold: float = 0.5) -> bool:
        """
        Check if two embeddings belong to the same person.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            threshold: Similarity threshold (default 0.5)
            
        Returns:
            True if same person, False otherwise
        """
        return self.compute_similarity(emb1, emb2) >= threshold


# Global instance for sharing across tracker
_face_recognizer: Optional[FaceRecognizer] = None


def get_face_recognizer(device: str = 'cuda') -> FaceRecognizer:
    """Get or create the global FaceRecognizer instance."""
    global _face_recognizer
    if _face_recognizer is None:
        _face_recognizer = FaceRecognizer(device=device)
    return _face_recognizer
