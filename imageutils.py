from pathlib import Path

import cv2
import numpy as np

from alignment_config import AlignmentConfig


def check_shape(arr: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    """Validate the shape of a numpy array (ignore -1)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if arr.ndim != len(shape):
        raise ValueError(
            f"{name} must be {len(shape)}D, got {arr.ndim}D"
        )
    for i, (actual, expected) in enumerate(zip(arr.shape, shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} has wrong size in dimension {i}: expected {expected},"
                f" got {actual}; full shape: {arr.shape}"
            )
    return arr


def check_bgr(img: np.ndarray, name: str) -> np.ndarray:
    """Validate that an image is in BGR format."""
    return check_shape(img, (-1, -1, 3), name)


def check_gray(img: np.ndarray, name: str) -> np.ndarray:
    """Validate that an image is in grayscale format."""
    return check_shape(img, (-1, -1), name)


def check_string(string: str, name: str) -> str:
    """Validate that an argument is a string."""
    if not isinstance(string, str):
        raise TypeError(f"{name} must be a string")
    return string


def check_extn(filename: str, name: str) -> str:
    """Validate that a filename ends with an image file extension."""
    allowed = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    if not filename.lower().endswith(allowed):
        raise ValueError(
            f"{name} must have a valid image file extension, got {filename}"
        )
    return filename


class ImageData:
    """
    A container class for storing an image and its associated
    attributes used for alignment.

    Attributes:
        image (np.ndarray | None): The image data as a numpy array
            (BGR). Defaults to None.
        filename (str | None): The name of the image file. Defaults to
            None.
        processed (np.ndarray | None): Processed version of the image
            used for keypoint detection (grayscale). Defaults to None.
        keyarea (np.ndarray | None): Rectangular area of interest
            represented as [x1, y1, x2, y2]. Defaults to None.
        kpts (np.ndarray | None): Array of keypoints that were manually
            selected by the user, or that correspond to detected
            features (shape: [N, 2]). Defaults to None.
        desc (np.ndarray | None): Array of descriptors that correspond
            to detected features (shape: [N, 256]). Defaults to None.
    """

    def __init__(
        self,
        image: np.ndarray | None = None,
        filename: str | None = None,
        processed: np.ndarray | None = None,
        keyarea: np.ndarray | None = None,
        kpts: np.ndarray | None = None,
        desc: np.ndarray | None = None
    ):
        """
        Initialize an ImageData object.

        Args:
            image (np.ndarray, optional): The image data as a numpy
                array (BGR). Defaults to None.
            filename (str, optional): The name of the image file.
                Defaults to None.
            processed (np.ndarray, optional): Processed version of the
                image used for keypoint detection (grayscale). Defaults
                to None.
            keyarea (np.ndarray, optional): Rectangular area of interest
                represented as [x1, y1, x2, y2]. Defaults to None.
            kpts (np.ndarray, optional): Array of keypoints that were
                manually selected by the user, or that correspond to
                detected features (shape: [N, 2]). Defaults to None.
            desc (np.ndarray, optional): Array of descriptors that
                correspond to detected features (shape: [N, 256]).
                Defaults to None.
        """
        self.image = image
        self.filename = filename
        self.processed = processed
        self.keyarea = keyarea
        self.kpts = kpts
        self.desc = desc

    # Getters and Setters ----------------------------------------------
    @property
    def image(self) -> np.ndarray | None:
        return self._image

    @image.setter
    def image(self, value: np.ndarray | None):
        if value is None:
            self._image = None
        else:
            self._image = check_bgr(value, "image")

    @property
    def filename(self) -> str | None:
        return self._filename

    @filename.setter
    def filename(self, value: str | None):
        if value is None:
            self._filename = None
        else:
            value = check_string(value, "filename")
            self._filename = check_extn(value, "filename")

    @property
    def processed(self) -> np.ndarray | None:
        return self._processed

    @processed.setter
    def processed(self, value: np.ndarray | None):
        if value is None:
            self._processed = None
        else:
            self._processed = check_gray(value, "processed")

    @property
    def keyarea(self) -> np.ndarray | None:
        return self._keyarea

    @keyarea.setter
    def keyarea(self, value: np.ndarray | None):
        if value is None:
            self._keyarea = None
        else:
            self._keyarea = check_shape(value, (4,), "keyarea")

    @property
    def kpts(self) -> np.ndarray | None:
        return self._kpts

    @kpts.setter
    def kpts(self, value: np.ndarray | None):
        if value is None:
            self._kpts = None
        else:
            self._kpts = check_shape(value, (-1, 2), "kpts")

    @property
    def desc(self) -> np.ndarray | None:
        return self._desc

    @desc.setter
    def desc(self, value: np.ndarray | None):
        if value is None:
            self._desc = None
        else:
            self._desc = check_shape(value, (-1, 256), "desc")

    # Public -----------------------------------------------------------
    def __str__(self):
        """Return the image filename, or an empty string if unset."""
        return self.filename if self.filename is not None else ""

    def process_image(self, cfg: AlignmentConfig):
        """
        Process this image for feature detection.

        The processed image is stored in the `processed` attribute of
        this object.

        Args:
            cfg (AlignmentConfig): Configuration specifying the
                preprocessing parameters.

        Returns:
            None: The processed image is stored in `self.processed`.
        """
        if self.image is None:
            raise ValueError("ImageData.image is None")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_tile_size, cfg.clahe_tile_size),
        )
        contrasted = clahe.apply(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            contrasted,
            (cfg.gaussian_ksize, cfg.gaussian_ksize),
            0,
        )
        self.processed = blurred


# Functions for np.ndarray images --------------------------------------
def scale_image(image: np.ndarray, scale: int | float) -> np.ndarray:
    """
    Return a rescaled version of the given image.

    Args:
        image (np.ndarray): The input image to be resized.
        scale (int | float): The scale factor for resizing. Must be > 0.

    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    # Validate scale
    if not isinstance(scale, (int, float)):
        raise TypeError("scale must be an int or float")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    # Validate image
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a np.ndarray")

    return cv2.resize(image, None, fx=scale, fy=scale)


def rgb_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image from BGR to RGB color space.

    Args:
        image (np.ndarray): The input image stored in BGR format
            (shape: [H, W, 3]).

    Returns:
        np.ndarray: The converted image in RGB format with the same
        shape as the input.
    """
    image = check_bgr(image, "image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Functions for ImageData objects --------------------------------------
def transform_image(
    image: ImageData,
    template: ImageData,
    keypairs: tuple,
    outlier_protection: bool = True
) -> ImageData:
    """
    Applies a homography transformation to an image based on given
    keypoint pairs.

    Args:
        image (ImageData): Source image to transform (must have `.image`
            set).
        template (ImageData): Destination image (must have `.image`
            set).
        keypairs (tuple): A tuple containing two numpy arrays
            (src_points, dst_points).
        outlier_protection (bool, optional): If True (default), applies
            robust estimation using the Least Median of Squares (LMEDS)
            method in OpenCV. If False, a least-squares approach is
            used.

    Returns:
        ImageData: The aligned/transformed image.
    """
    # Validate inputs
    if image.image is None:
        raise ValueError("image.image is None")
    if template.image is None:
        raise ValueError("template.image is None")
    if keypairs is None:
        raise ValueError("keypairs is None")
    if not isinstance(keypairs, tuple):
        raise TypeError("keypairs must be a tuple")
    if len(keypairs) != 2:
        raise ValueError(
            "keypairs must contain exactly two arrays: (src_points, dst_points)"
        )

    # Set method depending on outlier protection
    method = 0
    if outlier_protection:
        method = cv2.LMEDS

    # Compute the homography matrix
    H, mask = cv2.findHomography(
        keypairs[0],
        keypairs[1],
        method=method
    )
    # Use the homography matrix to transform the image
    h, w = template.image.shape[:2]
    aligned_raw = cv2.warpPerspective(image.image, H, (w, h))
    aligned = ImageData(aligned_raw)

    return aligned


def load_image(path_str: str) -> ImageData:
    """
    Load an image from disk and return it as an ImageData object.

    Args:
        path_str (str): Path to the image file. Must be a string with a
            supported image file extension.

    Returns:
        ImageData: The loaded image in BGR format (shape: [H, W, 3]).
    """
    # Validate the path
    path_str = check_string(path_str, "path_str")
    path_str = check_extn(path_str, "path_str")
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    raw = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if raw is None:
        raise OSError(f"Failed to read image: {path_str}")

    image = ImageData(raw, path.name)
    return image


def export_image(image: ImageData, out_dir: str | None) -> str | None:
    """
    Save an image to a specified directory, or to the default working
    directory if `out_dir` is None.

    Args:
        image (ImageData): The image to export.
        out_dir (str | None): Directory to save the image in. If None,
            the image is saved in the current working directory.

    Returns:
        str | None: The full path where the image was saved, or None if
        saving failed.
    """
    # Validate image
    if not isinstance(image, ImageData):
        raise TypeError("image must be an ImageData object")
    if image.image is None:
        raise ValueError("Provided image has not been loaded")
    if image.filename is None:
        raise ValueError("Provided image does not have a filename")

    # Determine save path
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / image.filename
    else:
        # Save in current working directory
        save_path = Path(image.filename)

    # Attempt to save image
    if cv2.imwrite(str(save_path), image.image):
        return str(save_path.resolve())
    else:
        return None
