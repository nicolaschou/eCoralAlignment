"""
========================================
File:        align_stack_superpoint.py
Description: This script aligns images using SuperPoint keypoint 
             detection on user-selected reference areas.
Author:      Nico Chou
========================================
"""


import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import os
from superpoint import superpoint_pytorch
import torch
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


def get_area_coords(
    image_data: ImageData,
    num_areas: int,
) -> list:
    """
    Collects rectangular areas in an image from user input.

    Args:
        image_data (ImageData): ImageData object for the image.
        num_areas (int): Number of areas to collect from the user.

    Returns:
        list: A list of tuples, each containing (x1, x2, y1, y2)  
              coordinates representing the selected rectangular areas.
    """
    plural = "s" if num_areas != 1 else ""

    # Plot image
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)
    ax.imshow(image_data.image)
    ax.set_title(
        f"Drag {num_areas} area{plural} around key shapes in the image "
        f"({image_data.filename})"
    )

    areas = []  # stores area coordinates
    boxes = []  # stores the rectangle patches

    # Callback for RectangleSelector; acts when a rectangle is drawn.
    def onselect(eclick, erelease):
        if len(areas) < num_areas:
            # Sort coordinates
            # Top left: (x1, y1), bottom right: (x2, y2)
            x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
            y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])

            # Draw the rectangle after it is selected
            box = patches.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                edgecolor="r",
                facecolor="none"
            )
            boxes.append(box)
            ax.add_patch(box)
            areas.append((x1, x2, y1, y2))
            fig.canvas.draw()

    # RectangleSelector allows the user to draw rectangles
    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels"
    )

    # Key press callback
    def onkey(event):
        # backspace -> remove last collected area
        if event.key == "backspace":
            if areas:
                box = boxes.pop()
                box.remove()
                areas.pop()
                fig.canvas.draw()

        # enter -> close figure (only after areas are collected)
        if event.key == "enter":
            if len(areas) == num_areas:
                rect_selector.set_active(False)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
            else:
                print(f"Drag {num_areas} area{plural} first")

    # Connect the event handler and display the image
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    return areas


def pre_process(
    image: np.ndarray,
    clip_limit=3,
    tile_size=4,
    ksize=27,
    debug=False
) -> np.ndarray:
    """
    Pre-processes an input image for feature detection.

    Args:
        image (np.ndarray): Input color image in RGB format.
        clip_limit (float): Threshold for contrast limiting in CLAHE. 
        tile_size (int): Size of grid tiles (tile_size x tile_size) 
                         used in CLAHE.
        ksize (int): Kernel size for Gaussian blur. Must be odd and > 0.

    Returns:
        np.ndarray: Processed image in grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    contrasted = clahe.apply(gray)

    # Apply a Gaussian blur
    blurred = cv2.GaussianBlur(contrasted, (ksize, ksize), 0)

    if debug:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(blurred, cmap="gray")
        plt.show()

    return blurred


def match_features(
    image: np.ndarray,
    template: np.ndarray,
    superpoint: superpoint_pytorch.SuperPoint,
    scale=0.5,
    ratio=0.8,
    debug=False
) -> tuple:
    """
    Detects and matches keypoint features between an input image and a 
    template image using SuperPoint.

    Args:
        image (np.ndarray): Input image in RGB format.
        template (np.ndarray): Template image in RGB format.
        superpoint (superpoint_pytorch.SuperPoint): Initialized 
                                                    SuperPoint model 
                                                    for keypoint 
                                                    detection.
        scale (float, optional): Scale factor for processing and 
                                 matching.
        ratio (float, optional): Lowe’s ratio-test threshold.
        debug (bool, optional): If True, displays a matplotlib figure 
                                showing the matched keypoints.

    Returns:
        tuple: (ptsI, ptsT)
            - ptsI (np.ndarray): Array of matched keypoint coordinates 
                                 from the input image (shape: [N, 2]).
            - ptsT (np.ndarray): Array of matched keypoint coordinates 
                                 from the template image 
                                 (shape: [N, 2]).
    """
    # Resize and pre-process images
    image_p = pre_process(image) / 255
    template_p = pre_process(template) / 255
    image_p = cv2.resize(image_p, None, fx=scale, fy=scale)
    template_p = cv2.resize(template_p, None, fx=scale, fy=scale)

    with torch.no_grad():
        # Convert image into tensor [1, 1, H, W]
        tensorI = torch.from_numpy(image_p[None, None]).float()
        tensorT = torch.from_numpy(template_p[None, None]).float()
        resultsI = superpoint({"image": tensorI})
        resultsT = superpoint({"image": tensorT})

    # Extract rescaled keypoints and descriptors
    kptsI = resultsI["keypoints"][0].cpu().numpy()
    descI = resultsI["descriptors"][0].cpu().numpy()
    kptsT = resultsT["keypoints"][0].cpu().numpy()
    descT = resultsT["descriptors"][0].cpu().numpy()

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(descI, descT, k=2)

    matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            matches.append(m)

    if debug:
        # Combine images side by side
        hI, wI = image_p.shape[:2]
        hT, wT = template_p.shape[:2]
        height = max(hI, hT)
        combined = np.zeros((height, wI + wT), dtype=np.uint8)
        combined[:hI, :wI] = image_p * 255
        combined[:hT, wI:wI + wT] = template_p * 255

        # Plot combined image
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(combined, cmap="gray")
        ax.set_title(f"Matched Keypoints ({len(matches)})")
        ax.axis("off")

        for m in matches:
            # Get match coordinates
            x1, y1 = kptsI[m.queryIdx]
            x2, y2 = kptsT[m.trainIdx]
            x2 += wI

            # Draw keypoints
            ax.scatter([x1], [y1], s=10, c="lime", marker="o")
            ax.scatter([x2], [y2], s=10, c="lime", marker="o")

            # Draw connecting line
            ax.plot([x1, x2], [y1, y2], c="red", linewidth=0.3)

        plt.show()

    # Match the coordinates of keypoint pairs
    ptsI = np.zeros((len(matches), 2), dtype="float")
    ptsT = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        # Scale back to original image size
        ptsI[i] = kptsI[m.queryIdx] / scale
        ptsT[i] = kptsT[m.trainIdx] / scale

    return ptsI, ptsT


def get_keypoint_pairs(
    img_data: ImageData,
    tmp_data: ImageData,
    areasI: list,
    areasT: list,
    superpoint: superpoint_pytorch.SuperPoint
) -> tuple:
    """
    Finds keypoint pairs between specified areas of an input image and 
    a template image.

    Args:
        img_data (ImageData): ImageData object for the input image.
        tmp_data (ImageData): ImageData object for the template image.
        areasI (list): List of rectangular area coordinates in the 
                       input image, where each area is a tuple 
                       (x1, x2, y1, y2).
        areasT (list): List of rectangular area coordinates in the 
                       template image, where each area is a tuple 
                       (x1, x2, y1, y2).
        superpoint (superpoint_pytorch.SuperPoint): Initialized 
                                                    SuperPoint model 
                                                    for keypoint 
                                                    detection.

    Returns:
        tuple:
            merged_kptsI (np.ndarray): Array of keypoint coordinates 
                                       from the input image 
                                       (shape: [N, 2])
            merged_kptsT (np.ndarray): Array of keypoint coordinates 
                                       from the template image 
                                       (shape: [N, 2])

    Raises:
        ValueError: If the number of areas in the input and template 
                    images do not match.
    """
    if len(areasI) != len(areasT):
        raise ValueError(
            "Number of areas in input and template images do not match."
        )

    all_kptsI = []
    all_kptsT = []

    # Find keypoint pairs for each area and translate accordingly
    for i in range(len(areasI)):
        x1I, x2I, y1I, y2I = areasI[i]
        x1T, x2T, y1T, y2T = areasT[i]

        # Extract only the areas of interest from the numpy arrays
        img_area = img_data.image[y1I:y2I, x1I:x2I]
        tmp_area = tmp_data.image[y1T:y2T, x1T:x2T]

        # Match features and transform the resulting keypoints
        kptsI, kptsT = match_features(img_area, tmp_area, superpoint)
        kptsI += np.array([x1I, y1I])
        kptsT += np.array([x1T, y1T])
        all_kptsI.append(kptsI)
        all_kptsT.append(kptsT)

    # Merge the sets of keypoints into one numpy array
    if all_kptsI:
        merged_kptsI = np.concatenate(all_kptsI, axis=0)
        merged_kptsT = np.concatenate(all_kptsT, axis=0)
    else:
        merged_kptsI = np.empty((0, 2))
        merged_kptsT = np.empty((0, 2))

    return merged_kptsI, merged_kptsT


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
        method=cv2.LMEDS
    )
    # Use the homography matrix to transform the image
    (h, w) = template_data.image.shape[:2]
    aligned = cv2.warpPerspective(image_data.image, H, (w, h))
    aligned_image_data = ImageData(
        aligned, f"aligned_{image_data.filename}")

    return aligned_image_data


def align_stack(num_areas) -> list:
    """
    Aligns a stack of images to a common reference frame using 
    feature-based matching.

    This function loads a series of images and iteratively aligns 
    each image in the stack to the previously aligned image using 
    user-selected reference areas and matched keypoint pairs. Each 
    aligned image is exported after transformation.

    Returns:
        list: A list of aligned ImageData objects, in the same order as 
              the input stack.
    """
    stack = load_images()
    if not stack:
        print(
            "No valid images were loaded. Please check your files and try "
            "again."
        )
        sys.exit(0)

    stack_areas = []
    aligned_images = []
    seed_images = []

    while True:
        seed_image = input(
            'Enter the filename of a seed image (maximum of 4) or type "done" '
            + 'to continue with alignment:\n'
        )
        seed_found = False
        if seed_image.strip() == "done":
            break
        if seed_image in seed_images:
            print(f"{seed_image} already selected as a seed image.\n")
            continue
        for i, image_data in enumerate(stack):
            if image_data.filename == seed_image:
                seed_found = True
                break
        if not seed_found:
            print(f"{seed_image} not found among uploaded images.\n")
            continue
        seed_images.append(seed_image)
        if len(seed_images) >= 4:
            break
    
    # Rebuild stack with the seed images first
    seed_stack = [img for img in stack if img.filename in seed_images]
    non_seed_stack = [img for img in stack if img.filename not in seed_images]
    stack = seed_stack + non_seed_stack

    # Obtain areas for keypoint detection
    for i in range(len(stack)):
        areas = get_area_coords(stack[i], num_areas)
        if len(areas) < num_areas:
            print(
                "Plot closed before all areas were selected. Exiting program."
            )
            sys.exit(0)
        stack_areas.append(areas)

    # Load SuperPoint model and set to evaluation mode
    superpoint = superpoint_pytorch.SuperPoint(
        nms_radius=5,
        detection_threshold=0.005,
        remove_borders=8,
    ).eval()

    # Load the pretrained weights from the .pth file
    weights = torch.load("superpoint/superpoint_v6_from_tf.pth")

    # Load the pretrained weights into the SuperPoint model
    superpoint.load_state_dict(weights)

    for i in range(len(stack)):
        print(f'Aligning "{stack[i].filename}"...')
        aligned = None
        if i == 0:
            aligned = ImageData(stack[i].image, f"aligned_{stack[i].filename}")
        else:
            # Collect keypoint pairs with previously aligned images
            # (at most 10)
            kptsI_sets = []
            kptsT_sets = []
            start_index = max(0, len(aligned_images) - 10)
            for j in range(start_index, len(aligned_images)):
                kptsI, kptsT = get_keypoint_pairs(
                    stack[i],
                    aligned_images[j],
                    stack_areas[i],
                    stack_areas[0],
                    superpoint
                )
                if kptsI.shape[0] == 0:
                    continue
                print(len(kptsI))
                kptsI_sets.append(kptsI)
                kptsT_sets.append(kptsT)

            if not kptsI_sets:
                print(f"No matches found for {stack[i].filename}; skipping.")
                continue

            # Concatenate all pairs
            kptsI_all = np.vstack(kptsI_sets)
            kptsT_all = np.vstack(kptsT_sets)
            keypairs = (kptsI_all, kptsT_all)

            aligned = transform_image(stack[i], stack[0], keypairs)

        # Store and export the aligned image
        aligned_images.append(aligned)
        export_image(aligned)

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
    align_stack(1)


if __name__ == "__main__":
    main()
