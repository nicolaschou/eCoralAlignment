from dataclasses import dataclass


@dataclass
class AlignmentConfig:
    """
    Configuration parameters for the SuperPoint image alignment pipeline
    implemented in superpoint_alignment.py.

    Attributes:
        weights_path (str): Path to the pretrained SuperPoint weights.
        aligned_prefix (str): Prefix added to filenames of aligned
            images.

        max_comparisons (int): Maximum number of template images to
            compare against when aligning a new unaligned image.

        clahe_clip_limit (float): Contrast-limited adaptive histogram
            equalization (CLAHE) clip limit for contrast enhancement.
        clahe_tile_size (int): Tile size used in CLAHE preprocessing.
        gaussian_ksize (int): Kernel size for Gaussian blurring. Must
            be an odd integer.

        superpoint_nms_radius (int): Non-maximum suppression radius for
            keypoint detection in SuperPoint.
        superpoint_det_thresh (float): Detection threshold for
            SuperPoint keypoint confidence.
        superpoint_remove_borders (int): Number of pixels to ignore
            along image borders when detecting keypoints.
        ratio_test (float): Ratio threshold used during descriptor
            matching (Lowe’s ratio test).
        scale (float): Scaling factor applied to images before
            processing with SuperPoint.

        debug (bool): If True, enables additional visualization and
            diagnostic output for debugging.
    """
    # File management
    weights_path: str = "superpoint/superpoint_v6_from_tf.pth"
    aligned_prefix: str = "aligned_"

    # Alignment pipeline
    max_comparisons: int = 15

    # Preprocessing
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 4
    gaussian_ksize: int = 27  # must be odd

    # Matching
    superpoint_nms_radius: int = 5
    superpoint_det_thresh: float = 0.005
    superpoint_remove_borders: int = 8
    ratio_test: float = 0.8
    scale: float = 0.5

    # Debug
    debug: bool = False
