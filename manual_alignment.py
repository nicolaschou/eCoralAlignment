import sys

import tkinter as tk

from fileui import AlignmentManager
from imageui import get_keypoints, template_plot
import imageutils as iu


def manual_alignment() -> list:
    """
    Perform manual alignment on selected images with user-selected
    keypoints.

    This function launches the AlignmentManager GUI for user selection
    of unaligned images, template images, an output directory, and the
    number of keypoints to select. For the template and each unaligned
    image, the user selects corresponding keypoints. From these keypoint
    pairs, the function estimates the geometric transform between each
    image and the template, and saves the aligned result to the
    specified output directory.

    Returns:
        list: ImageData objects corresponding to the aligned versions of
        the provided images.
    """
    aligned_prefix = "aligned_"
    unaligned, templates, out_dir, num_points = run_alignment_manager()
    template = templates[0]
    results = []

    # Collect keypoints for the template image
    template.keypoints = get_keypoints(template, num_points)
    if template.keypoints is None:
        print(
            "Plot closed before keypoints were selected. Exiting program."
        )
        sys.exit(0)
    
    # Align each image
    for image in unaligned:
        tmp_fig, tmp_ax = template_plot(template, template.keypoints)
        image.keypoints = get_keypoints(image, num_points, tmp_fig)
        if image.keypoints is None:
            print(
                "Plot closed before keypoints were selected. Exiting program."
            )
            sys.exit(0)
        keypairs = (image.keypoints, template.keypoints)
        aligned = iu.transform_image(image, template, keypairs, False)
        aligned.filename = f"{aligned_prefix}{image.filename}"

        # Store and export the aligned image
        results.append(aligned)
        iu.export_image(aligned, out_dir)

    return results


def run_alignment_manager() -> tuple:
    """
    Retrieve images and details required to execute manual alignment
    using the AlignmentManager GUI.
    """
    root = tk.Tk()
    root.title("Alignment Manager")

    # Only one template is used for manual alignment
    manager = AlignmentManager(root, 1, True)

    manager.pack(fill="both", expand=True)
    root.wait_window(root)
    results = getattr(root, "results", None)
    if results is None:
        print("Alignment Manager closed unexpectedly. Exiting program.")
        sys.exit(0)

    # Return templates and unaligned images and set output directory
    unaligned = results["unaligned"]
    templates = results["templates"]
    out_dir = results["out_dir"]
    num_points = results["num_points"]
    return unaligned, templates, out_dir, num_points


if __name__ == "__main__":
    manual_alignment()