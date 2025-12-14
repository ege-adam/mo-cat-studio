"""
Model Factory - Sets up SAM 3D Body estimator with optional components.
This is custom glue code, NOT part of the original sam_3d_body library.
"""
import torch

from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """
    Set up SAM 3D Body estimator with optional components.

    Args:
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use (default: "vitdet")
        segmentor_name: Name of segmentor to use (default: "sam2")
        fov_name: Name of FOV estimator to use (default: "moge2")
        detector_path: URL or path for human detector model
        segmentor_path: Path to human segmentor model (optional)
        fov_path: path for FOV estimator
        device: Device to use (default: auto-detect cuda/cpu)

    Returns:
        estimator: SAM3DBodyEstimator instance ready for inference
    """
    print(f"Loading SAM 3D Body model from {hf_repo_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load core model from HuggingFace
    model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    # Initialize optional components
    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        try:
            from tools.build_detector import HumanDetector
            human_detector = HumanDetector(name=detector_name, device=device)
        except ImportError:
            print(f"Warning: Could not import HumanDetector.")

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        try:
            from tools.build_sam import HumanSegmentor
            human_segmentor = HumanSegmentor(
                name=segmentor_name, device=device, path=segmentor_path
            )
        except ImportError:
            print(f"Warning: Could not import HumanSegmentor.")

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        try:
            from tools.build_fov_estimator import FOVEstimator
            fov_estimator = FOVEstimator(name=fov_name, device=device)
        except ImportError:
            print(f"Warning: Could not import FOVEstimator.")

    # Create estimator wrapper
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator
