"""
Author: Nico Chou

Semi-automated image alignment pipeline using SuperPoint feature
detection.
"""
import sys
import tkinter as tk
import re
import time
from tkinter import filedialog
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import torch

from alignment_config import AlignmentConfig
from fileui import AlignmentManager
from imageui import get_keyarea, show_debug
import imageutils as iu
from imageutils import ImageData
from superpoint import superpoint_pytorch


@dataclass
class AlignmentJob:
    unaligned: list = None
    templates: list = None
    out_dir: str = None


def superpoint_alignment_job(job: AlignmentJob, cfg: AlignmentConfig) -> list:
    """
    Align a series of unaligned images to one or more template images
    using SuperPoint keypoint detection and descriptor matching.

    For each unaligned image, feature matches are computed between it
    and up to a specified maximum of the preceding templates in the
    processing order (feature detection is restricted to keyareas in
    each image that are selected by the user). From these feature
    matches, the function estimates the geometric transform, saves the
    aligned result to the specified output directory, and stores it as a
    template to be used for aligning the following images. If no
    templates are provided, the first unaligned image becomes the
    initial template.

    Args:
        job (AlignmentJob): Container holding the unaligned images,
            initial template images, and output directory path for the
            alignment job.
        cfg (AlignmentConfig): Configuration object containing model
            parameters and debug settings.

    Returns:
        list: ImageData objects corresponding to the aligned versions of
        the provided images.
    """

    # Load templates and unaligned images
    unaligned = job.unaligned
    templates = job.templates
    out_dir = job.out_dir    

    # Initialize a SuperPoint model
    superpoint = configure_superpoint(cfg)

    # Store initial number of templates
    num_temps = len(templates)

    for i in range(len(unaligned)):
        print(f'Aligning "{unaligned[i].filename}"...')
        aligned = None

        if i == 0 and not templates:
            # If there are no templates, consider first image as aligned
            aligned = ImageData(
                image=unaligned[i].image,
                filename=f"{cfg.aligned_prefix}{unaligned[i].filename}",
                keyarea=unaligned[i].keyarea
            )
        else:
            # Collect keypoint pairs with templates
            kptsI_sets = []
            kptsT_sets = []
            start_index = max(0, len(templates) - cfg.max_comparisons)
            for j in range(start_index, len(templates)):
                kptsI, kptsT = get_keypoint_pairs(
                    unaligned[i],
                    templates[j],
                    superpoint,
                    cfg
                )
                if kptsI.shape[0] == 0:
                    continue
                kptsI_sets.append(kptsI)
                kptsT_sets.append(kptsT)

            if not kptsI_sets:
                print(f"No matches for {unaligned[i].filename}; skipping.")
                continue

            # Concatenate all pairs
            kptsI_all = np.vstack(kptsI_sets)
            if len(kptsI_all) < 4:
                print(f"<4 matches for {unaligned[i].filename}; skipping.")
                continue
            kptsT_all = np.vstack(kptsT_sets)
            keypairs = (kptsI_all, kptsT_all)

            # Transform/align
            template = templates[0]
            aligned = iu.transform_image(unaligned[i], template, keypairs, cfg)
            aligned.filename = f"{cfg.aligned_prefix}{unaligned[i].filename}"

            # The keyarea of the aligned image is now the same as in the
            # template(s) it was aligned to
            aligned.keyarea = template.keyarea

        # Store and export the aligned image
        templates.append(aligned)
        iu.export_image(aligned, out_dir)

    # Return list of newly aligned images
    return templates[num_temps:]


def parse_files(cfg: AlignmentConfig) -> list[AlignmentJob]:
    """
    Build alignment jobs.

    Images are grouped by image ID (the filename prefix). For each image
    ID, the function loads up to cfg.max_comparisons of the most recent
    template images from the corresponding <image_id>_superpoint_aligned
    folder and matches them with all new images sharing that ID.
    Template images must contain a date in YYYY-MM-DD format in their
    filename.

    Args:
        cfg (AlignmentConfig): Configuration object containing model
            parameters and debug settings.

    Returns:
        list: A list of alignment jobs.
    """
    folders = select_folders()
    new_images_dir = Path(folders["New Images Folder"])
    templates_dir = Path(folders["Templates Folder"])

    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

    new_img_paths = sorted(
        p for p in new_images_dir.iterdir()
        if p.is_file() and _is_image_file(p)
    )

    a_jobs = {}

    for img_path in new_img_paths:
        # e.g. 4a-2026-04-13.JPG -> 4a
        image_id = img_path.stem.split("-", 1)[0]

        if image_id not in a_jobs:
            # e.g. 4a -> 4a_superpoint_aligned
            template_folder = templates_dir / f"{image_id}_superpoint_aligned"

            if not template_folder.is_dir():
                print(f"No templates found for {img_path.name}; skipping.")
                continue

            dated_tmp_paths = []

            # Get the dates of templates
            for tmp_path in template_folder.iterdir():
                if (
                    not tmp_path.is_file()
                    or not _is_image_file(tmp_path)
                ):
                    continue

                match = date_pattern.search(tmp_path.name)
                if match is None:
                    continue

                dated_tmp_paths.append((match.group(), tmp_path))

            sorted_tmp_paths = sorted(
                dated_tmp_paths,
                key=lambda item: item[0],
                reverse=True
            )

            # Select the most recent templates up to cfg.max_comparisons
            tmp_paths = [
                path for date, path in sorted_tmp_paths[:cfg.max_comparisons]
            ]

            a_jobs[image_id] = AlignmentJob(
                unaligned=[],
                templates=[],
                out_dir=str(template_folder)
            )

            for tmp_path in tmp_paths:
                a_jobs[image_id].templates.append(iu.load_image(str(tmp_path)))

        a_jobs[image_id].unaligned.append(iu.load_image(str(img_path)))

    return list(a_jobs.values())


def select_folders():
    """Prompt selection of the new image and template directories."""
    root = tk.Tk()
    root.withdraw()

    folders = {
        "New Images Folder": None,
        "Templates Folder": None,
    }

    for key in folders.keys():
        path = filedialog.askdirectory(title=key)

        if not path:
            print("Folder selection cancelled. Exiting program.")
            sys.exit(0)

        folders[key] = path

    return folders


def _is_image_file(path: Path) -> bool:
    """Determine whether a file has a supported image extension."""
    try:
        iu.check_extn(path.name, path.name)
        return True
    except ValueError:
        return False


def set_keyareas(images: list):
    """
    Get keyareas for the provided images and store them in their
    associated ImageData objects.
    """
    for image in images:
        keyarea = get_keyarea(image)
        if keyarea is None:
            print(
                "Plot closed before an area was selected. Exiting program."
            )
            sys.exit(0)
        image.keyarea = keyarea


def configure_superpoint(
    cfg: AlignmentConfig
) -> superpoint_pytorch.SuperPoint:
    """
    Configure and return a SuperPoint model with the provided
    parameters.
    """
    # Load SuperPoint model and set to evaluation mode
    superpoint = superpoint_pytorch.SuperPoint(
        nms_radius=cfg.superpoint_nms_radius,
        detection_threshold=cfg.superpoint_det_thresh,
        remove_borders=cfg.superpoint_remove_borders
    ).eval()

    # Load weights into the SuperPoint model
    weights = torch.load(cfg.weights_path)
    superpoint.load_state_dict(weights)

    return superpoint


def get_keypoint_pairs(
    image: ImageData,
    template: ImageData,
    superpoint: superpoint_pytorch.SuperPoint,
    cfg: AlignmentConfig
):
    """
    Compute matching keypoint pairs between an image and its template
    using SuperPoint descriptors and a ratio-based matching criterion.

    This function performs nearest-neighbor matching using a brute-force
    matcher (L2 norm), and applies a ratio test to filter matches (a 
    match is kept only if its distance is smaller than a fixed ratio
    multiplied by that of the next-best match). If debugging is enabled,
    it visualizes the matched keypoints.

    Args:
        image (ImageData): The unaligned image.
        template (ImageData): The template image.
        superpoint (superpoint_pytorch.SuperPoint): Initialized
            SuperPoint model used for keypoint detection and descriptor
            extraction.
        cfg (AlignmentConfig): Configuration object containing model
            parameters and debug settings.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - `kptsI` (ndarray): Array of shape [N, 2] with (x, y)
              coordinates of matched keypoints in `image`.
            - `kptsT` (ndarray): Array of shape [N, 2] with (x, y)
              coordinates of corresponding keypoints in `template`.
    """
    process_features(image, superpoint, cfg)
    process_features(template, superpoint, cfg)

    # Find two best matches for each descriptor
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(image.desc, template.desc, k=2)

    # Keep best match if its distance is less than a specified ratio of
    # the next best match's distance
    matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < cfg.ratio_test * n.distance:
            matches.append(m)

    # Match the coordinates of keypoint pairs
    kptsI = np.zeros((len(matches), 2), dtype="float")
    kptsT = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        kptsI[i] = image.kpts[m.queryIdx]
        kptsT[i] = template.kpts[m.trainIdx]

    if cfg.debug:
        show_debug(image, template, kptsI, kptsT)

    return kptsI, kptsT


def process_features(
    image: ImageData,
    superpoint: superpoint_pytorch.SuperPoint,
    cfg: AlignmentConfig
):
    """
    Process image, detect SuperPoint features (descriptors associated
    with keypoint coordinates) within a specified key area, and store
    the results.

    Args:
        image (ImageData): ImageData object that contains the source
            image. Processing and feature detection results are stored
            in this object.
        superpoint (superpoint_pytorch.SuperPoint): Initialized
            SuperPoint model.
        cfg (AlignmentConfig): Parameters for preprocessing and feature
            detection.
    """
    if image.processed is None:
        image.process_image(cfg)

    x1, y1, x2, y2 = image.keyarea
    if image.kpts is None or image.desc is None:

        # Detect features in keyarea
        img_area = image.processed[y1:y2, x1:x2] / 255.0  # normalize
        area_scaled = iu.scale_image(img_area, cfg.scale)
        with torch.no_grad():
            # Convert np.ndarray into tensor [1, 1, H, W]
            tensor = torch.from_numpy(area_scaled[None, None]).float()
            results = superpoint({"image": tensor})

        # Extract keypoints and descriptors
        kpts = results["keypoints"][0].cpu().numpy()
        kpts_rescaled = kpts / cfg.scale
        kpts_rescaled += np.array([x1, y1])
        image.kpts = kpts_rescaled
        image.desc = results["descriptors"][0].cpu().numpy()


def run_alignment_job(args):
    """Execute a single alignment job."""
    job, cfg = args
    return superpoint_alignment_job(job, cfg)


if __name__ == "__main__":
    cfg = AlignmentConfig()
    a_jobs = parse_files(cfg)

    for job in a_jobs:
        if job.templates:
            set_keyareas([job.templates[0]])
            for temp in job.templates[1:]:
                temp.keyarea = job.templates[0].keyarea

        if job.unaligned:
            set_keyareas(job.unaligned)
    
    start = time.time()

    with Pool(processes=min(2, len(a_jobs))) as pool:
        results = pool.map(
            run_alignment_job,
            [(job, cfg) for job in a_jobs]
        )
    
    end = time.time()
    print("Runtime:", end - start, "seconds")
