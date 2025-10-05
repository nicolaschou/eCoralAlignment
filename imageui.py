import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from mpl_interactions import zoom_factory
import imageutils as iu
from imageutils import ImageData
    

def mpl_window(img: np.ndarray, maxdim: int) -> tuple:
    """
    Create a matplotlib window sized to fit the given image.

    Args:
        img (np.ndarray): The image as a numpy array (BGR or grayscale).
        maxdim (int): Maximum dimension of the figure.

    Returns:
        tuple:
            - fig (matplotlib.figure.Figure): The Matplotlib figure.
            - ax (matplotlib.axes.Axes): The axes object.
    """
    h, w = img.shape[:2]
    dpi = max(h, w) / maxdim
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    return fig, ax


def get_keyarea(image_data: ImageData) -> np.ndarray:
    """
    Collect a rectangular area in an image from user input.

    The area is drawn with the mouse, can be reset with Backspace, and
    is confirmed with Enter. Once confirmed, the coordinates of the
    rectangle are returned.

    Args:
        image_data (ImageData): ImageData object for the image.

    Returns:
        np.ndarray: A numpy array containing (x1, y1, x2, y2)
            coordinates representing the selected area.
    """

    # Plot image
    rgb = iu.rgb_image(image_data.image)
    fig, ax = mpl_window(rgb, 10)
    ax.imshow(rgb)
    ax.set_title(image_data.filename)

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
                print(f"Select an area first")

    # Connect the event handler and display the image
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    return area


def get_keypoints(
    image_data: ImageData,
    num_points: int,
    tmp_fig=None,
) -> np.ndarray:
    """
    Displays an image and collects a specified number of (x, y) points 
    clicked by the user.

    Keypoints are selected and drawn on clicks. Users can remove the
    last point with Backspace and finish selection with Enter (only
    after the required number of points is collected).

    Args:
        image_data (ImageData): ImageData object for the image.
        num_points (int): Number of points to collect.
        tmp_fig (matplotlib.figure.Figure): Figure containing the 
            template and its keypoints.

    Returns:
        np.ndarray: Array of shape (num_points, 2) with clicked (x, y) 
            coordinates in the original dimensions.
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
    fig, ax = mpl_window(rgb, 10)
    ax.imshow(rgb)
    zoom_factory(ax)
    ax.set_title(
        f"Click {num_points} keypoint{plural} on the image "
        f"({image_data.filename})"
    )

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
                print(f"Click {num_points} point{plural} first")

    # Connect the event handler and display the image
    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    # Scale points to original dimensions and return
    orig_points = np.array(points) / scale
    return orig_points