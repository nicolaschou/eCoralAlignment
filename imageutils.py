import numpy as np

class ImageData:
    """
    A container class for storing an image and its filename.

    Attributes:
        image (np.ndarray): The image data as a numpy array.
        filename (str): The name of the image file.
    """

    def __init__(
            self, 
            image: np.ndarray,
            filename: str,
            prefix = ""
            preprocessed = None,
            aligned = None,
            keypoints = None,
            keyareas = None
        ):
        """
        Initializes an ImageData object.

        Args:
            image (np.ndarray): The image data as a numpy array.
            filename (str): The name of the image file.
        """
        self.image = image
        self.filename = filename