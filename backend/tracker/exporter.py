import numpy as np
import os
from typing import List, Dict, Optional

# Try to import pymomentum for native FBX export
try:
    import pymomentum.geometry as pym_geometry
    MOMENTUM_ENABLED = True
except ImportError:
    MOMENTUM_ENABLED = False
    print("[MHRExporter] pymomentum not available, FBX export disabled")

class MHRExporter:
    """
    Exports MHR track data to FBX format using pymomentum's native export.
    
    This uses the same skeleton and character that the MHR model uses,
    ensuring perfect compatibility with the animation data.
    """
    
    def __init__(self, mhr_model=None):
        """
        Initialize exporter.
        
        Args:
            mhr_model: Optional MHR model instance to get character from.
                      If None, will load fresh from assets.
        """
        self.character = None
        self.num_joints = 127  # MHR has 127 joints
        
        if MOMENTUM_ENABLED:
            try:
                if mhr_model is not None and hasattr(mhr_model, 'character'):
                    # Use character from the model directly
                    self.character = mhr_model.character
                    print(f"[MHRExporter] Using character from MHR model")
                else:
                    # Try to load from default asset path
                    from mhr.io import get_default_asset_folder, get_mhr_fbx_path, get_mhr_model_path
                    folder = get_default_asset_folder()
                    fbx_path = get_mhr_fbx_path(folder, lod=1)
                    model_path = get_mhr_model_path(folder)
                    
                    if os.path.exists(fbx_path) and os.path.exists(model_path):
                        self.character = pym_geometry.Character.load_fbx(
                            fbx_path, model_path, load_blendshapes=False
                        )
                        print(f"[MHRExporter] Loaded character from {fbx_path}")
                
                if self.character:
                    self.num_joints = self.character.skeleton.size
                    print(f"[MHRExporter] Character has {self.num_joints} joints")
                    
            except Exception as e:
                print(f"[MHRExporter] Failed to load character: {e}")
                self.character = None

    def export_to_fbx(self, 
                      mhr_params_history: List[np.ndarray], 
                      output_path: str,
                      fps: float = 30.0,
                      cam_t_history: List[np.ndarray] = None,
                      rot_history: List[np.ndarray] = None):
        """
        Export track history to GLB file using pymomentum's GLTF export.
        
        Note: Despite the function name, this now exports to GLB format because
        pymomentum was built without the Autodesk FBX SDK.
        
        Uses mhr_model_params directly which contain all the pose information
        in the correct format for the MHR character.
        
        Args:
            mhr_params_history: List of MHR model parameters per frame [N_frames x N_params]
                               Each parameter array contains: [global_trans*10, global_rot, body_pose, hand_pose, scales]
            output_path: Path to save the GLB file
            fps: Frames per second for animation
            cam_t_history: List of camera translations per frame [N_frames x 3] for root motion
            rot_history: (unused, for backward compatibility)
        """
        if not MOMENTUM_ENABLED:
            print("[MHRExporter] pymomentum not available, cannot export GLB")
            return False
        
        if not mhr_params_history or len(mhr_params_history) == 0:
            print("[MHRExporter] No MHR parameters to export")
            return False
        
        if self.character is None:
            print("[MHRExporter] No character loaded, cannot export GLB")
            return False
        
        try:
            # Stack all frames into a single array [n_frames x n_params]
            joint_params = np.stack(mhr_params_history, axis=0).astype(np.float32)

            n_frames = joint_params.shape[0]
            n_params = joint_params.shape[1]

            # Validate data - check for NaN or inf values
            if np.any(np.isnan(joint_params)):
                print(f"[MHRExporter] ERROR: Motion data contains NaN values")
                return False
            if np.any(np.isinf(joint_params)):
                print(f"[MHRExporter] ERROR: Motion data contains Inf values")
                return False
            
            # Inject root translation from cam_t_history if provided
            # The first 3 params in MHR are global_trans * 10
            if cam_t_history is not None and len(cam_t_history) == n_frames:
                print(f"[MHRExporter] Injecting root translation from cam_t_history")
                for i, cam_t in enumerate(cam_t_history):
                    if cam_t is not None:
                        # cam_t is in meters, MHR expects global_trans * 10
                        # Also flip Y and Z for coordinate system conversion
                        joint_params[i, 0] = cam_t[0] * 10  # X
                        joint_params[i, 1] = -cam_t[1] * 10  # Y (flip)
                        joint_params[i, 2] = -cam_t[2] * 10  # Z (flip)
            
            print(f"[MHRExporter] Exporting {n_frames} frames with {n_params} parameters per frame")
            
            # Change extension to .glb
            glb_path = output_path.replace('.fbx', '.glb')

            print(f"[MHRExporter] Calling pymomentum Character.save()...")
            print(f"[MHRExporter]   path: {glb_path}")
            print(f"[MHRExporter]   motion shape: {joint_params.shape}")
            print(f"[MHRExporter]   motion dtype: {joint_params.dtype}")
            print(f"[MHRExporter]   motion range: [{joint_params.min():.2f}, {joint_params.max():.2f}]")

            # Use pymomentum's unified save method which supports GLB
            # This works without the Autodesk FBX SDK
            try:
                pym_geometry.Character.save(
                    path=glb_path,
                    character=self.character,
                    fps=fps,
                    motion=joint_params,  # Just the numpy array, not a tuple
                    options=pym_geometry.FileSaveOptions(
                        mesh=True,
                        locators=True,
                        collisions=False,
                        blend_shapes=True
                    )
                )
            except Exception as save_error:
                print(f"[MHRExporter] ERROR during pymomentum save: {save_error}")
                import traceback
                traceback.print_exc()
                return False

            # Verify file was created and has valid size
            if not os.path.exists(glb_path):
                print(f"[MHRExporter] ERROR: GLB file not created at {glb_path}")
                return False

            file_size = os.path.getsize(glb_path)
            if file_size < 1000:  # GLB should be at least 1KB
                print(f"[MHRExporter] ERROR: GLB file too small ({file_size} bytes), likely corrupted")
                return False

            # Validate GLB file format and JSON content
            # GLB files must start with magic bytes: "glTF" (0x676C5446) followed by version (2) and length
            try:
                import json
                import struct

                with open(glb_path, 'rb') as f:
                    header = f.read(12)
                    if len(header) < 12:
                        print(f"[MHRExporter] ERROR: GLB file header too short")
                        return False

                    # Check magic bytes
                    magic = header[0:4]
                    if magic != b'glTF':
                        print(f"[MHRExporter] ERROR: Invalid GLB magic bytes: {magic!r} (expected b'glTF')")
                        return False

                    # Check version (should be 2)
                    version = int.from_bytes(header[4:8], byteorder='little')
                    if version != 2:
                        print(f"[MHRExporter] WARNING: GLB version is {version}, expected 2")

                    # Check length matches file size
                    declared_length = int.from_bytes(header[8:12], byteorder='little')
                    if declared_length != file_size:
                        print(f"[MHRExporter] ERROR: GLB declared length ({declared_length}) doesn't match file size ({file_size})")
                        return False

                    # Read first chunk (should be JSON)
                    chunk_header = f.read(8)
                    if len(chunk_header) < 8:
                        print(f"[MHRExporter] ERROR: GLB chunk header too short")
                        return False

                    chunk_length = struct.unpack('<I', chunk_header[0:4])[0]
                    chunk_type = chunk_header[4:8]

                    if chunk_type != b'JSON':
                        print(f"[MHRExporter] ERROR: First chunk is not JSON: {chunk_type!r}")
                        return False

                    # Read and validate JSON chunk
                    json_data = f.read(chunk_length)
                    try:
                        gltf_json = json.loads(json_data)

                        # Check for required "asset" field (GLTF 2.0 spec requirement)
                        if 'asset' not in gltf_json:
                            print(f"[MHRExporter] ERROR: GLTF JSON missing required 'asset' field")
                            print(f"[MHRExporter] JSON keys present: {list(gltf_json.keys())}")
                            return False

                        # Validate asset has version
                        if 'version' not in gltf_json['asset']:
                            print(f"[MHRExporter] ERROR: GLTF asset missing 'version' field")
                            return False

                        print(f"[MHRExporter] GLTF validation passed - asset.version: {gltf_json['asset']['version']}")

                    except json.JSONDecodeError as je:
                        print(f"[MHRExporter] ERROR: Invalid JSON in GLB: {je}")
                        return False

            except Exception as e:
                print(f"[MHRExporter] ERROR: Failed to validate GLB format: {e}")
                import traceback
                traceback.print_exc()
                return False

            print(f"[MHRExporter] GLB exported successfully to: {glb_path} ({file_size} bytes, valid format)")
            return True
            
        except Exception as e:
            print(f"[MHRExporter] GLB export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


