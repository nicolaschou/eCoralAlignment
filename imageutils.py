import numpy as np
import cv2
from pathlib import Path


def check_bgr(img: np.ndarray, name: str) -> np.ndarray:
    """Validates that an image is in BGR format."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be stored as a numpy array")
    elif img.ndim == 3 and img.shape[2] == 3:
        return img
    else:
        raise ValueError(
            f"{name} must be BGR (shape: [H, W, 3]), got {img.shape}"
        )


def check_gray(img: np.ndarray, name: str) -> np.ndarray:
    """Validates that an image is in grayscale format."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be stored as a numpy array")
    elif img.ndim == 2:
        return img
    else:
        raise ValueError(
            f"{name} must be grayscale (shape: [H, W]), got {img.shape}"
        )


def check_string(string: str, name: str) -> str:
    """Validates that an argument is a string."""
    if not isinstance(string, str):
        raise TypeError(f"{name} must be a string")
    else:
        return string
    

def check_shape(arr: np.ndarray, dim1: int, name: str) -> np.ndarray:
    """Validates that an array has shape [N, dim1]."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if arr.ndim != 2 or arr.shape[1] != dim1:
        raise ValueError(
            f"{name} must have shape [N, {dim1}], got {arr.shape}"
        )
    return arr


def check_extn(filename: str, name: str) -> str:
    """Validates that a filename ends with an image file extension."""
    allowed = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
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
        aligned (np.ndarray | None): Aligned version of the image 
            (BGR). Defaults to None.
        keyareas (np.ndarray | None): Array of rectangular areas of
            interest represented as [x1, y1, x2, y2] (shape: [N, 4]).
            Defaults to None.
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
            aligned: np.ndarray | None = None,
            keyareas: np.ndarray | None = None,
            kpts: np.ndarray | None = None,
            desc: np.ndarray | None = None
        ):
        """
        Initializes an ImageData object.

        Args:
            image (np.ndarray, optional): The image data as a numpy
                array (BGR). Defaults to None.
            filename (str, optional): The name of the image file.
                Defaults to None
            processed (np.ndarray, optional): Processed version of the
                image used for keypoint detection (grayscale). Defaults
                to None.
            aligned (np.ndarray, optional): Aligned version of the
                image (BGR). Defaults to None.
            keyareas (np.ndarray, optional): Array of rectangular areas
                of interest represented as [x1, y1, x2, y2] (shape: 
                [N, 4]). Defaults to None.
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
        self.aligned = aligned
        self.keyareas = keyareas
        self.kpts = kpts
        self.desc = desc


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
    def aligned(self) -> np.ndarray | None:
        return self._aligned


    @aligned.setter
    def aligned(self, value: np.ndarray | None):
        if value is None:
            self._aligned = None
        else:
            self._aligned = check_bgr(value, "aligned")


    @property
    def keyareas(self) -> np.ndarray | None:
        return self._keyareas


    @keyareas.setter
    def keyareas(self, value: np.ndarray | None):
        if value is None:
            self._keyareas = None
        else:
            self._keyareas = check_shape(value, 4, "keyareas")


    @property
    def kpts(self) -> np.ndarray | None:
        return self._kpts


    @kpts.setter
    def kpts(self, value: np.ndarray | None):
        if value is None:
            self._kpts = None
        else:
            self._kpts = check_shape(value, 2, "kpts")


    @property
    def desc(self) -> np.ndarray | None:
        return self._desc


    @desc.setter
    def desc(self, value: np.ndarray | None):
        if value is None:
            self._desc = None
        else:
            self._desc = check_shape(value, 256, "desc")


def load_image(path_str: str):
    """
    Loads an image from disk and returns it as a numpy array.

    Args:
        path_str (str): Path to the image file. Must be a string with a
            supported image file extension.

    Returns:
        np.ndarray: The loaded image in BGR format (shape: [H, W, 3]).

    Raises:
        TypeError: If `path_str` is not a string.
        ValueError: If `path_str` does not have a valid image extension.
        FileNotFoundError: If the specified file does not exist.
        OSError: If the file exists but cannot be read as an image.
    """
    # Validate the path
    if path_str is None:
        raise TypeError("path_str must be a string")
    path_str = check_string(path_str, "path_str")
    path_str = check_extn(path_str, "path_str")
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    
    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read image: {path_str}")
    return image


def export_image(image: np.ndarray, filename: str, out_dir: str) -> str | None:
    """
    Saves an image (stored as a numpy array) to a specified directory.

    Args:
        image (np.ndarray): The image to export in BGR format.
        filename (str): The filename of the exported image. This must
            end in a valid image file extension (e.g. .jpg). The
            extension dictates the file format of the exported image.
        out_dir (str): The path of the directory to save the image to.

    Returns:
        str | None: The full path where the image was saved, or None if
        saving failed.

    Raises:
        TypeError: If any argument is of incorrect data type.
        ValueError: If any argument has invalid structure.
    """
    # Ensure specified output directory exists
    if out_dir is None:
        raise TypeError("out_dir must be a string")
    out_dir = check_string(out_dir, "out_dir")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate filename
    if filename is None:
        raise TypeError("filename must be a string")
    filename = check_string(filename, "filename")
    filename = check_extn(filename, "filename")
    
    # Ensure image is a numpy array in the correct format
    if image is None:
        raise TypeError("image must be a numpy array")
    image = check_bgr(image, "image")
    
    save_path = out_path / filename
    if cv2.imwrite(str(save_path), image):
        return str(save_path)
    else:
        return None