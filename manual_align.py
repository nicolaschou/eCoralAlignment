"""
========================================
File:        manual_align.py
Description: This script aligns images using a homography matrix 
             computed from user-selected keypoint pairs.
Author:      Nico Chou
========================================
"""


import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from mpl_interactions import zoom_factory
import matplotlib
import matplotlib.pyplot as plt
import os
import sys


class ImageData:
    """
    A container class for storing an image and its filename.

    Attributes:
        image (np.ndarray): The image data as a numpy array.
        filename (str): The name of the image file.
    """

    def __init__(self, image: np.ndarray, filename: str):
        """
        Initializes an ImageData object.

        Args:
            image (np.ndarray): The image data as a numpy array.
            filename (str): The name of the image file.
        """
        self.image = image
        self.filename = filename


def load_images() -> list:
    """
    Opens a file dialog for the user to select multiple image files and loads 
    the images as ImageData objects into a list. Images are added to the list 
    in the order they are selected in the file dialog.

    Returns:
        list: A list of ImageData objects containing the images in RGB 
              format selected by the user.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()

    # Ask user to select multiple files
    file_paths = list(filedialog.askopenfilenames(title="Select image files"))
    root.update()
    root.destroy()

    # Extract images from their paths
    images = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read {file_path}")
            continue
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(file_path)
        images.append(ImageData(image, filename))

    return images


def get_keypoints(
    image_data: ImageData,
    num_points: int,
    tmp_fig=None,
    scale=0.3
) -> np.ndarray:
    """
    Displays an image and collects a specified number of (x, y) points 
    clicked by the user.

    Args:
        image_data (ImageData): ImageData object for the image.
        num_points (int): Number of points to collect.
        tmp_fig (matplotlib.figure.Figure): Figure containing the 
                                            template and its keypoints.
        scale (float): How large the displayed image is relative to the 
                       original.

    Returns:
        np.ndarray: Array of shape (num_points, 2) with clicked (x, y) 
                    coordinates in the original dimensions.
    """
    plural = "s" if num_points != 1 else ""

    # Plot resized image with zoom functionality
    image_scaled = cv2.resize(image_data.image, None, fx=scale, fy=scale)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)
    ax.imshow(image_scaled)
    zoom_factory(ax)
    ax.set_title(
        f"Click {num_points} keypoint{plural} on the image "
        f"({image_data.filename})"
    )

    points = []  # stores keypoint coordinates
    dots = []  # stores the Line2D dot objects

    # Mouse click callback
    def onclick(event):
        # If a tool is selected, do nothing
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return

        # Plot dot at last mouse position when no tool is selected
        if event.xdata is not None and event.ydata is not None:
            if len(points) < num_points:  # only record up to num_points
                x, y = event.xdata, event.ydata
                points.append((x, y))
                dot, = ax.plot(x, y, "ro", ms=3)  # unpack
                dots.append(dot)
                fig.canvas.draw()

    # Key press callback
    def onkey(event):
        # backspace -> remove last clicked point
        if event.key == "backspace":
            if dots:
                dot = dots.pop()
                dot.remove()
                points.pop()
                fig.canvas.draw()

        # enter -> close figures (only after all points are collected)
        if event.key == "enter":
            if len(points) == num_points:
                fig.canvas.mpl_disconnect(cid_click)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
                if tmp_fig is not None:
                    plt.close(tmp_fig)
            else:
                print(f"Click {num_points} point{plural} first")

    # Connect the event handler and display the image
    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    # Scale points to original dimensions and return
    orig_points = np.array(points) / scale
    return orig_points


def get_template_window(
    template_data: ImageData,
    keypoints: np.ndarray,
    scale=0.3
) -> matplotlib.figure.Figure:
    """
    Creates a matplotlib figure displaying a template with annotated 
    keypoints.

    Args:
        template_data (ImageData): The template image.
        keypoints (np.ndarray): Array of keypoints (shape: [N, 2]).
        scale (float): How large the displayed image is relative to the 
                       original.

    Returns:
        matplotlib.figure.Figure: The annotated figure.
    """
    # Scale image and display using matplotlib with scroll zoom enabled
    image_scaled = cv2.resize(template_data.image, None, fx=scale, fy=scale)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
    ax.imshow(image_scaled)
    zoom_factory(ax)
    ax.set_title(f"Template w/ Keypoints ({template_data.filename})")
    ax.axis("off")

    # Plot and number points
    scaled_points = keypoints * scale
    text_offset = image_scaled.shape[1] / 40
    ax.plot(scaled_points[:, 0], scaled_points[:, 1], "ro", ms=6)
    for i, (x, y) in enumerate(scaled_points):
        ax.text(
            x + text_offset,
            y,
            str(i+1),
            color="orange",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold"
        )

    return fig


def transform_image(
    image_data: ImageData,
    template_data: ImageData,
    keypairs: list
) -> ImageData:
    """
    Applies a homography transformation to an image based on given 
    keypoint pairs.

    Args:
        image_data (ImageData): Image to be transformed.
        template_data (ImageData): Destination image.
        keypairs (list): A list containing two numpy arrays 
                         [src_points, dst_points].

    Returns:
        ImageData: The aligned/transformed image as an ImageData object.
    """
    # Compute the homography matrix
    (H, mask) = cv2.findHomography(
        keypairs[0], 
        keypairs[1], 
        method=0
    )
    # Use the homography matrix to transform the image
    (h, w) = template_data.image.shape[:2]
    aligned = cv2.warpPerspective(image_data.image, H, (w, h))
    aligned_image_data = ImageData(
        aligned,
        f"aligned_{image_data.filename}"
    )

    return aligned_image_data


def user_input(stack: list) -> tuple:
    """
    Prompts the user to select a template image filename from the 
    provided stack and specify the number of keypoints to collect per 
    image.

    Args:
        stack (list): List of ImageData objects which should contain 
                      the template image.

    Returns:
        tuple: (template_index, num_points)
            - template_index (int): Index of the selected template 
              image in the stack, or -1 if the user quits.
            - num_points (int): Number of keypoints to collect per 
              image (>=4), or -1 if the user quits.
    """
    # Collect the index of the template image in the stack
    template_index = None
    while True:
        template = input(
            'Please enter the filename of the template image (this should '
            'match one of the files you previously selected) or type "quit" '
            'to end the program:\n'
        )
        if template.strip() == "quit":
            return (-1, -1)
        for i, image_data in enumerate(stack):
            if image_data.filename == template:
                template_index = i
                break
        if template_index is None:
            print(f"Could not find {template} among the selected files.\n")
        else:
            break

    # Collect the number of keypoint pairs to collect
    num_points = None
    while True:
        num_points = input(
            'Enter the number of keypoints to collect per image (must be >= '
            '4) or type "quit" to end the program:\n'
        ).strip()
        if num_points == "quit":
            return (-1, -1)
        if num_points.isdigit():
            num_points = int(num_points)
            if num_points >= 4:
                break
            else:
                print("Number of keypoints must be >= 4.\n")
        else:
            print("Please enter a valid integer.\n")

    return template_index, num_points


def export_image(image_data: ImageData):
    """
    Exports the image and filename specified in an ImageData object to 
    disk.

    Args:
        image_data (ImageData): ImageData object containing the image.
    """
    image_bgr = cv2.cvtColor(image_data.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_data.filename, image_bgr)


def align_stack() -> list:
    """
    Loads images, prompts for a template image and number of keypoints, 
    collects keypoints, aligns images to the template and exports.

    Returns:
        list: The aligned images as ImageData objects.
    """
    stack = load_images()
    if not stack:
        print(
            "No valid images were loaded. Please check your files and try "
            "again."
        )
        sys.exit(0)
    
    template_index, num_points = user_input(stack)
    kpts_stack = []
    aligned_images = []

    # Check if user chose to close the program
    if template_index == -1:
        sys.exit(0)

    # Move the template image to the first index
    if template_index != 0:
        template_image = stack.pop(template_index)
        stack.insert(0, template_image)
    template_data = stack[0]

    # Collect keypoints and export aligned images
    for i, image_data in enumerate(stack):
        if i == 0:
            # No need to align the template image
            keypoints = get_keypoints(image_data, num_points)
            if len(keypoints) < num_points:
                print(
                    "Plot closed before all points were selected. " +
                    "Exiting program."
                )
                sys.exit(0)
            kpts_stack.append(keypoints)
            aligned_images.append(image_data)
        else:
            # Only display the template after the first iteration
            tmp_fig = get_template_window(template_data, kpts_stack[0])
            keypoints = get_keypoints(image_data, num_points, tmp_fig)
            if len(keypoints) < num_points:
                print(
                    "Plot closed before all points were selected. " +
                    "Exiting program."
                )
                sys.exit(0)
            kpts_stack.append(keypoints)
            # Align the image with the template and export
            keypairs = [kpts_stack[i], kpts_stack[0]]
            aligned_image_data = transform_image(
                image_data,
                template_data,
                keypairs
            )
            aligned_images.append(aligned_image_data)
            export_image(aligned_image_data)

    return aligned_images


def main():
    align_stack()


if __name__ == "__main__":
    main()
