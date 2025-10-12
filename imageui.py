"""
Author: Nico Chou

Matplotlib-based visualization and input utilities for interactive image
alignment (area selection, keypoints, and debugging).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.widgets import RectangleSelector
from mpl_interactions import zoom_factory

import imageutils as iu
from imageutils import ImageData

mpl.rcParams['toolbar'] = 'None'
mpl.use("QtAgg") 


def mpl_window(img: np.ndarray, maxdim: int | float) -> tuple:
    """
    Create a matplotlib window sized to fit the given image.

    Args:
        img (np.ndarray): The image as a numpy array (BGR or grayscale).
        maxdim (int | float): Maximum dimension of the figure (inches).

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            - `fig` (matplotlib.figure.Figure): The Matplotlib figure.
            - `ax` (matplotlib.axes.Axes): The axes object.
    """
    h, w = img.shape[:2]
    dpi = max(h, w) / maxdim
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    return fig, ax


def template_plot(
    template_data: ImageData,
    keypoints: np.ndarray,
) -> tuple:
    """
    Create a matplotlib figure displaying a template with annotated
    keypoints.

    Args:
        template_data (ImageData): The template image.
        keypoints (np.ndarray): Array of keypoints (shape: [N, 2]).

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            - `fig` (matplotlib.figure.Figure): The Matplotlib figure.
            - `ax` (matplotlib.axes.Axes): The axes object.
    """
    # Rescale to 1500 pixels to reduce zoom lag
    rgb = iu.rgb_image(template_data.image)
    h, w = rgb.shape[:2]
    scale = 1
    if h > 1500 or w > 1500:
        scale = 1500/max(h, w)
        rgb = iu.scale_image(rgb, scale)

    # Plot resized image with zoom functionality
    fig, ax = mpl_window(rgb, 7)
    fig.canvas.manager.set_window_title(
        f"Template w/ Keypoints ({template_data.filename})"
    )
    ax.imshow(rgb)
    zoom_factory(ax)

    # Plot points
    scaled_points = keypoints * scale
    ax.plot(scaled_points[:, 0], scaled_points[:, 1], "ro", ms=3)

    # Label points
    for i, (x, y) in enumerate(scaled_points, start=1):
        ax.annotate(
            str(i),
            xy=(x, y),
            xycoords="data",
            xytext=(4, 10),
            textcoords="offset points",
            ha="left",
            va="top",
            color="orange",
            fontsize=10,
            weight="bold",
            clip_on=True,
            zorder=4,
        )

    return fig, ax


def get_keyarea(image_data: ImageData) -> np.ndarray | None:
    """
    Collect a rectangular area in an image from user input.

    The area is drawn with the mouse, can be reset with Backspace, and
    is confirmed with Enter. Once confirmed, the coordinates of the
    rectangle are returned.

    Args:
        image_data (ImageData): ImageData object for the image.

    Returns:
        np.ndarray | None: (x1, y1, x2, y2) integer coordinates, or
            None if the window is closed before an area is selected.
    """

    # Plot image
    rgb = iu.rgb_image(image_data.image)
    fig, ax = mpl_window(rgb, 9)
    fig.canvas.manager.set_window_title(image_data.filename)
    ax.imshow(rgb)

    area = None  # stores area coordinates
    box = None  # stores the rectangle patch

    # Callback for RectangleSelector; acts when a rectangle is drawn.
    def onselect(eclick, erelease):
        nonlocal area, box

        if area is None:
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
            ax.add_patch(box)
            area = np.array([x1, y1, x2, y2])
            fig.canvas.draw()

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
        nonlocal area, box

        # backspace -> reset area
        if event.key == "backspace":
            if area is not None:
                box.remove()
                box = None
                area = None
                fig.canvas.draw()

        # enter -> close figure (only after areas are collected)
        if event.key == "enter":
            if area is not None:
                rect_selector.set_active(False)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
            else:
                print("Select an area first")

    # Connect the event handler and display the image
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    return area


def get_keypoints(
    image_data: ImageData,
    num_points: int,
    tmp_fig: mpl.figure.Figure | None = None
) -> np.ndarray | None:
    """
    Displays an image and collects a specified number of (x, y) points
    clicked by the user.

    Keypoints are selected and drawn on clicks. Users can remove the
    last point with Backspace and finish selection with Enter (only
    after the required number of points is collected).

    Args:
        image_data (ImageData): ImageData object for the image.
        num_points (int): Number of points to collect.
        tmp_fig (matplotlib.figure.Figure | None, optional): Figure 
            containing the template and its keypoints. Defaults to None.

    Returns:
        np.ndarray | None: Array of shape (num_points, 2) with clicked
            (x, y) coordinates in the original dimensions, or None if
            the window is closed before all keypoints are selected.
    """
    plural = "s" if num_points != 1 else ""

    # Rescale to 1500 pixels to reduce zoom lag
    rgb = iu.rgb_image(image_data.image)
    h, w = rgb.shape[:2]
    scale = 1
    if h > 1500 or w > 1500:
        scale = 1500/max(h, w)
        rgb = iu.scale_image(rgb, scale)

    # Plot resized image with zoom functionality
    fig, ax = mpl_window(rgb, 9)
    fig.canvas.manager.set_window_title(
        f"Select {num_points} keypoint{plural} ({image_data.filename})"
    )
    ax.imshow(rgb)
    zoom_factory(ax)

    points = []  # stores keypoint coordinates
    dots = []  # stores the Line2D dot objects

    # Mouse click callback
    def onclick(event):
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
                print(f"Click {num_points} keypoint{plural} first")

    # Connect the event handler and display the image
    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    # Return None if the user closed the window early
    if not points or len(points) != num_points:
        if tmp_fig is not None:
            plt.close(tmp_fig)
        return None

    # Scale points to original dimensions and return
    orig_points = np.array(points) / scale
    return orig_points


def show_debug(
    image: ImageData,
    template: ImageData,
    kptsI: np.ndarray,
    kptsT: np.ndarray
):
    """
    Display a visual debug window showing matched keypoints between two
    images.

    Args:
        image (ImageData): The source image.
        template (ImageData): The template image.
        kptsI (np.ndarray): Array of shape [N, 2] containing (x, y)
            coordinates of keypoints in the source image.
        kptsT (np.ndarray): Array of shape [N, 2] containing (x, y)
            coordinates of corresponding keypoints in the template
            image.
    """
    # Combine images side by side
    hI, wI = image.image.shape[:2]
    hT, wT = template.image.shape[:2]
    height = max(hI, hT)
    combined = np.zeros((height, wI + wT, 3), dtype=np.uint8)
    combined[:hI, :wI, :] = iu.rgb_image(image.image)
    combined[:hT, wI:wI + wT, :] = iu.rgb_image(template.image)

    # Plot combined image
    fig, ax = mpl_window(combined, 12)
    fig.canvas.manager.set_window_title(
        f"{image.filename} vs. {template.filename} (# Matches: {len(kptsI)})"
    )
    ax.imshow(combined)
    ax.axis("off")

    # Plot matches
    for i in range(len(kptsI)):
        # Get match coordinates
        x1, y1 = kptsI[i]
        x2, y2 = kptsT[i]
        x2 += wI

        # Draw keypoints
        ax.scatter([x1], [y1], s=10, c="lime", marker="o")
        ax.scatter([x2], [y2], s=10, c="lime", marker="o")

        # Draw connecting line
        ax.plot([x1, x2], [y1, y2], c="red", linewidth=0.3)

    plt.show()
