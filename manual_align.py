"""
========================================
File:        manual_align.py
Description: This script aligns images using a homography matrix 
             computed from and user-selected key point pairs.
Author:      Nico Chou
Created:     7/23/2025
Last Edited: 8/4/2025
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


class ImageData:
    def __init__(self, image: np.ndarray, filename: str):
        self.image = image
        self.filename = filename


def load_images() -> list:
    """
    Opens a file dialog for the user to select multiple image files.

    Returns:
        list: A list of ImageData objects containing the images 
              selected by the user.
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
        tmp_fig (matplotlib.figure.Figure): template with the first 
                                            points chosen in the stack.
        scale (float): How large the displayed image is relative to the 
                       original.

    Returns:
        np.ndarray: Array of shape (num_points, 2) with clicked (x, y) 
                    coordinates in the original dimensions.
    """
    plural = 's' if num_points != 1 else ''

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
    
    points = [] # stores keypoint coordinates
    dots = [] # stores the Line2D dot objects

    # Mouse click callback
    def onclick(event):
        # If a tool is selected, do nothing
        toolbar = event.canvas.toolbar
        if toolbar is not None and toolbar.mode != '':
            return
        
        # Plot dot at last mouse position when no tool is selected
        if event.xdata is not None and event.ydata is not None:
            if len(points) < num_points: # only record up to num_points
                x, y = event.xdata, event.ydata
                points.append((x, y))
                dot, = ax.plot(x, y, 'ro', ms=3) # unpack
                dots.append(dot)
                fig.canvas.draw()

    # Key press callback
    def onkey(event):
        # backspace -> remove last clicked point
        if event.key == 'backspace':
            if dots:
                dot = dots.pop()
                dot.remove()
                points.pop()
                fig.canvas.draw()
        
        # enter -> close figures (only after all points are collected)
        if event.key == 'enter':
            if len(points) == num_points:
                fig.canvas.mpl_disconnect(cid_click)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
                plt.close(tmp_fig)
            else:
                print(f"Click {num_points} point{plural} first")

    # Connect the event handler and display the image
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    # Wait until window is closed to return
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    orig_points = np.array(points) / scale
    return orig_points


def transform_image(
        image_data: ImageData, 
        keypairs: list
    ) -> ImageData:
    """
    Applies a homography transformation to an image based on given 
    keypoint pairs.

    Args:
        image_data (ImageData): Image to be transformed.
        keypairs (list): A list of numpy arrays with format 
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
    (h, w) = image_data.image.shape[:2]
    aligned = cv2.warpPerspective(image_data.image, H, (w, h))
    aligned_image_data = ImageData(aligned, f"aligned_{image_data.filename}")
    return aligned_image_data


def get_template_window(
        image_data: ImageData, 
        keypoints: np.ndarray, 
        scale=0.3
    ) -> matplotlib.figure.Figure:
    """
    Creates a matplotlib figure displaying a template with annotated 
    keypoints.

    Args:
        image_data (np.ndarray): The template image.
        keypoints (np.ndarray): Array of ordered points.
        scale (float): How large the displayed image is relative to the 
                       original.

    Returns:
        matplotlib.figure.Figure: The annotated figure.
    """
    image_scaled = cv2.resize(image_data.image, None, fx=scale, fy=scale)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
    ax.imshow(image_scaled)
    zoom_factory(ax)
    ax.set_title(f"Template w/ Keypoints ({image_data.filename})")
    ax.axis('off')

    # Plot and number points
    scaled_points = keypoints * scale
    text_offset = image_scaled.shape[1] / 40
    ax.plot(scaled_points[:, 0], scaled_points[:, 1], 'ro', ms=6)
    for i, (x, y) in enumerate(scaled_points):
        ax.text(
            x+text_offset, 
            y, 
            str(i+1), 
            color='orange', 
            fontsize=10, 
            ha='center', 
            va='center', 
            weight='bold'
        )
    return fig    


def align_stack() -> list:
    """
    Loads images, prompts for a template image, collects keypoints, 
    aligns images to the template and exports.

    Returns:
        list: The aligned images as ImageData objects.
    """
    stack = load_images()
    template = input("Enter the filename of the template image (this should "
                     "match one of the files you previously selected):\n")
    kpts_stack = []
    aligned_images = []
    
    # Find the template image and move it to the first index
    template_index = None
    for i, image_data in enumerate(stack):
        if image_data.filename == template:
            template_index = i
    if template_index is None:
        raise FileNotFoundError(
            f"Could not find {template} among the selected files"
        )
    if template_index != 0:
        template_image = stack.pop(template_index)
        stack.insert(0, template_image)

    # Collect keypoints and export aligned images
    for i, image_data in enumerate(stack):
        # Only display the key after the first iteration
        if i == 0:
            keypoints = get_keypoints(image_data, 5)
            kpts_stack.append(keypoints)
            aligned_images.append(image_data)
        else:
            tmp_fig = get_template_window(stack[0], kpts_stack[0])
            keypoints = get_keypoints(image_data, 5, tmp_fig)
            kpts_stack.append(keypoints)
            keypairs = [kpts_stack[i], kpts_stack[0]]
            aligned_image_data = transform_image(image_data, keypairs)
            aligned_images.append(aligned_image_data)
            export_image(aligned_image_data)

    return aligned_images


def export_image(image_data: ImageData):
    """
    Saves the image and filename specified in an ImageData object to 
    disk.

    Args:
        image_data (ImageData): ImageData object containing the image.
    """
    image_bgr = cv2.cvtColor(image_data.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_data.filename, image_bgr)


def main():
    align_stack()

if __name__ == "__main__":
    main()