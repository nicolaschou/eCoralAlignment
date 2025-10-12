# eCoralAlignment

### Python utilities for aligning images to a common reference frame:

  - `manual_alignment.py` — ***manual keypoint selection:*** you select a set of corresponding keypoints for each image; the script computes a homography matrix from the selected keypoints and warps each image to the designated template.
  - `superpoint_alignment.py` — ***semi-automated keypoint detection:*** you draw one or more rectangular regions of interest on each image; SuperPoint detects kepoints within those areas, and the script aligns each image to the rest of the stack by comparing with up to a preset number of previously aligned images.
    - **Relies on Rémi Pautrat and Paul-Édouard Sarlin’s [PyTorch implementation of the SuperPoint model](https://github.com/rpautrat/SuperPoint), originally proposed by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich in ["SuperPoint: Self-Supervised Interest Point Detection and Description"](https://arxiv.org/abs/1712.07629).** Their code and the associated pretrained weights are included in `superpoint/`. See citations at the end of this markdown file.

These utilities were developed to align underwater images of coral to measure their growth for a research project.


## Requirements

  - Python 3.9+
  - Matplotlib, mpl-interactions, NumPy, OpenCV, PyQt5 or PyQt6, PyTorch 

    `pip install matplotlib mpl-interactions numpy opencv-python PyQt6 torch`


## `manual_alignment.py` Usage

`python manual_alignment.py`

### Alignment Manager GUI
  - **Template Selection:** Click *"Add Images"* under the *Templates* panel and select the corresponding file. Only one template image file will be accepted—all other images will be aligned to this image's reference frame.
  - **Unaligned Image Selection:** Click *"Add Images"* under the *Unaligned* panel and select the corresponding files. Unaligned images can be selected in multiple batches and there is no limit on how many.
  - **Output Folder Selection:** Click *"Select Output Folder"* and choose the desired output location. Verify that the folder you chose matches the name displayed in the window.
  - **Specify Number of Keypoints:** Enter the number of keypoints that you would like to select for each image in the *"Number of Keypoints"* field. This value must be at least 4.
  - **Image Removal:** Click *"Remove Selected Image"* under the *Unaligned* or *Templates* panel to discard the selected image in the respective panel. Removed images will not be considered for alignment.
  - **Proceed to Keypoint Selection:** Click *"Done"* to begin keypoint selection.

### Keypoint Selection
  - **Zoom Functionality:** Scroll to zoom in and out of the displayed image for more precise keypoint selection.
  - **Template Features:** The first image displayed is your template image. Select your specified number of keypoints by clicking on the image. For the best performance, select stationary features (sharp corners, distinct markings, etc.). These should be visible in all of the selected images and spread out in 3D space.
  - **Unaligned Image Features:** The images displayed after the template are unaligned. Click on the same features you chose for the template, in the exact same order. A reference window showing the template and its selected keypoints is provided.
  - **Point Deletion:** Press "delete" on your keyboard to remove the last point you selected.
  - **Confirmation:** Press "return" or "enter" on your keyboard once you have accurately selected the appropriate number of keypoints. The program will only proceed to the next image if all keypoints have been selected. You cannot undo this action.

### Important Notes
  - Output images are transformed to **match the dimensions of the template image**.
  - If no output folder is selected, output images are exported to the working directory.
  - If the Alignment Manager window is closed without using the *"Done"* button, **the program quits**.
  - If an image display window is closed before the appropriate number of keypoints are selected, **the program quits**.
  - Aligned images are named **"aligned_{original filename}"**.
  - Images may not be displayed at full resolution. Their largest dimension is limited to 1500 pixels to prevent lag during zooming.
  - Output images are exported directly after their keypoints have been selected.


## `superpoint_alignment.py` Usage

`python superpoint_alignment.py`

### Alignment Process
  1. **Image Selection:** Unaligned images and template images (images that are already aligned) are selected.
  2. **Order Selection:** Images and templates are arranged in the desired processing order (all templates precede unaligned images).
  3. **Keyarea Selection:** For each image and template, a keyarea is selected — the specific region where feature detection and matching will occur, ideally a stationary subject in the image.
  4. **Alignment:** (Do the following for every unaligned image)
      - **Feature Detection:** Use SuperPoint to detect features between it and a specified number of preceding images in the processing order, restricted to the selected keyareas.
      - **Feature Matching:** For each pair (unaligned and previously aligned), perform **Lowe's ratio test**. That is, for each feature in the unaligned image, the two best matches are found in the previously aligned image; a match is accepted only if the distance to the best match is smaller than a fixed ratio (default 0.8) times that of the next best match.
      - **Homography Transformation:** Concatenate all match sets into a single group and estimate the homography transformation from the combined matches. Apply the homography transformation to the image, then replace the unaligned image in the processing order with the aligned result.
      
      ***All preceding images in the processing order have already been aligned to the same reference frame. Aligning to multiple previously aligned images increases robustness against poor matching between any single pair of images.***

### Alignment Manager GUI
  - **Template Selection:** Click *"Add Images"* under the *Templates* panel and select the corresponding files. You may select as many templates as the maximum number of comparisons to be made in the processing order.
  - **Unaligned Image Selection:** Click *"Add Images"* under the *Unaligned* panel and select the corresponding files. Unaligned images can be selected in multiple batches and there is no limit on how many.
  - **Reordering Images:** In the *Unaligned* and *Templates* panels, drag and drop files into the desired processing order. For the best performance, images that are the most similar should be adjacent.
  - **Output Folder Selection:** Click *"Select Output Folder"* and choose the desired output location. Verify that the folder you chose matches the name displayed in the window.
  - **Image Removal:** Click *"Remove Selected Image"* under the *Unaligned* or *Templates* panel to discard the selected image in the respective panel. Removed images will not be considered for alignment.
  - **Proceed to Keyarea Selection:** Click *"Done"* to begin keyarea selection.

### Keyarea Selection
  - **Area Selection:** Click and drag on the image to select the keyarea. For the best performance, select a stationary subject with clear features (sharp corners, distinct markings, etc.). The same subject must be selected in all images.
  - **Area Deletion:** Press "delete" on your keyboard to remove the area you selected.
  - **Confirmation:** Press "return" or "enter" on your keyboard once you have accurately selected the keyarea. The program will only proceed to the next image if a keyarea has been selected. You cannot undo this action.

### Alignment Parameters
  - Alignment parameters are stored in the `AlignmentConfig` dataclass (implemented in `alignment_config.py`).
  - To update the default parameters, simply update them in this file.
  - A few key insights:
    - A large `gaussian_ksize` improves alignment when images are dissimilar in fine detail.
    - `scale` significantly impacts compute time.
    - `superpoint_nms_radius` and `scale` are heavily interdependent. As `scale` increases, the effective non-maximum suppression radius decreases because the image is analyzed at a larger resolution, reducing the relative size of the suppression area.

### Important Notes
  - Output images are transformed to **match the dimensions of the first image in the processing order**.
  - **The keyarea selected for the first displayed image is reused for all newly aligned images.**
  - If no output folder is selected, output images are exported to the working directory.
  - If the Alignment Manager window is closed without using the *"Done"* button, **the program quits**.
  - If an image display window is closed before a keyarea is selected, **the program quits**.
  - Aligned images are named **"aligned_{original filename}"** by default.
  - Output images are exported directly after their feature detection iteration is complete.


## Citation

The `superpoint_alignment.py` pipeline relies Rémi Pautrat and Paul-Édouard Sarlin’s [PyTorch implementation of the SuperPoint model](https://github.com/rpautrat/SuperPoint), originally proposed by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich in ["SuperPoint: Self-Supervised Interest Point Detection and Description"](https://arxiv.org/abs/1712.07629).

See [`superpoint/`](./superpoint) and [`superpoint/SUPERPOINT.`](./superpoint/SUPERPOINT.md) for information about the exact files used.


BibTeX citation for "SuperPoint: Self-Supervised Interest Point Detection and Description":
```
@inproceedings{detone18superpoint,
  author    = {Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  booktitle = {CVPR Deep Learning for Visual SLAM Workshop},
  year      = {2018},
  url       = {http://arxiv.org/abs/1712.07629}
}
```