"""
.. versionadded:: 0.2
.. codeauthor:: Aman Gajendra Jain <aj161198@gmail.com>

**External Dependencies**

.. hlist::
   :columns: 4

   - cv2
   - numpy
   - matplotlib
   - keras

**Internal Dependencies**

.. hlist::
   :columns: 2
    :private-members:

   - :class:`~quark.experimental.opencvutility`
   - :class:`~quark.helpers.fileconstants`

Automatic GCP detector is an **experimental** module that uses a combination
of computer vision principles and machine-learning to detect if a GCP (L-Shape)
is present in an image.

**Recommendations**

- The Geo-Information is not used anywhere, hence, most of the hard-coded
  numbers are based on experimentation and trials, which had to be satisfied
  across multiple-datasets, which resulted in giving lot of false-detections,
  to improvise it the first and the fore-most task to be done, is use
  Geo-Information at all places where checks are introduced namely,

  - contour_area
  - box_area
  - cv2.countNonZero(edges) <= 200, by changing it with calculating the
    perimeter of GCP

- Automation of RGB-threshold value based on image-intensity

- Improvisation in ML-model, the model at present is trained with only 804
  positives and more will be needed to train and tune hyper-parameters

- Improvisation in the tolerances added to the bounding-boxes

- Improvisation in Edge-Detection Algorithm

- Improvisation in peak-detection and peak-verification algorithm
"""

# Copyright (C) 2018 Skylark Drones

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from opencvutility import (
    rotate_jpeg,
    morphology,
    extract_contours,
    check_concavity,
    extract_roi,
)

# Kernel for Morphological operations
kernel = np.ones((3, 3), np.uint8)


def classifier(img):
    """
    Determines the probability of a binary image being a GCP or not using
    ML-model. It takes 3.3% of the total time

    :param numpy.ndarray img: A binary image containing edges
    :return: Probability of being and not being a GCP
    """
    prediction = load_model(
        os.path.join(FileConstants().QUARK_LIB_DIR, 'experimental', 'model.h5')
    )
    img = np.array(img).reshape([1, 28, 28, 1])
    probability = prediction.predict(img.reshape([1, 28, 28, 1]))

    return probability


def _rgb_threshold(rgb, rgb_t=180):
    """
    Function to do color thresholding in an RGB-colorspace with the threshold
    value of (rgb_t, rgb_t, rgb_t)

    :param numpy.ndarray rgb: RGB-Image
    :param int rgb_t: RGB-threshold value
    :return: Binary Image
    """
    low_rgb = np.array([rgb_t, rgb_t, rgb_t])
    high_rgb = np.array([255, 255, 255])
    black_white = cv2.inRange(rgb, low_rgb, high_rgb)
    return black_white


def _drgb_threshold(img, drgb_t=30):
    """
    Function to do thresholding in Differential-RGB-colorspace with threshold
    value of (drgb_t, drgb_t, drgb_t)

    .. figure:: ../_images/drgb.jpg
       :align: center

    .. note::
        This process takes lot of time, around 90% of the total-time and improvements could be made by using
        PCA

    :param numpy.ndarray img: RGB-Image
    :param int drgb_t: DRGB-Threshold
    :return: Binary Image
    """
    blank_drgb = np.zeros(img.shape, np.uint8)
    # h, w, c = img.shape

    b = np.array(img[:, :, 0], np.int)
    g = np.array(img[:, :, 1], np.int)
    r = np.array(img[:, :, 2], np.int)
    blank_drgb[:, :, 0] = np.absolute(np.subtract(b, r))
    blank_drgb[:, :, 1] = np.absolute(np.subtract(g, b))
    blank_drgb[:, :, 2] = np.absolute(np.subtract(r, g))
    lower = np.array([0, 0, 0])
    higher = np.array([drgb_t, drgb_t, drgb_t])
    mask = cv2.inRange(blank_drgb, lower, higher)
    return mask


def _check_contour_area(contours):
    """
    Checks for contour area and append contours to a list i.e. 0 <= area <= 850
    pixels

    .. note::
        The upper limits and the lower limits are hard-coded and could be
        improvised by geo-information

    :param list contours: List of all contours
    :return: List of all contours that satisifies the area check
    """
    area = []
    for contour in contours:
        if 0 <= cv2.contourArea(contour) <= 850:
            area.append(contour)
    return area


def _box_area(contours, img):
    """
    Checks for bounding-box area, aspect ratio and adds tolerance to the
    bounding-box

    .. note::
        The limits of area could be improvised from Geo-Information
        The tolerance of +- 10 is not efficient enough to get the whole
        edge-image correctly, improvements are needed

    :param list contours: List of contours
    :param numpy.ndarray img: RGB-Image
    :return: List of bounding-box
    """
    bbox = []
    height, width, channels = img.shape
    for contour in contours:
        x, y, h, w = cv2.boundingRect(contour)
        if 50 <= h * w <= 1500 and (abs(h - w) <= max(h / 2, w / 2)):
            bbox.append(
                (
                    max(x - 10, 0),
                    max(y - 10, 0),
                    min(h + 20, height - y + 10),
                    min(w + 20, width - x + 10),
                )
            )
    return bbox


def _extract_angles(dxs, dys):
    """
    Calculates orientation of edge from derivative in x and y direction

    :param list dxs: List of dx
    :param list dys: List of dy
    :return: List of angles in range (0, 360)
    """
    angles = []
    for (dx, dy) in zip(dxs, dys):
        if dx > 0 and dy >= 0:
            angle = np.arctan(dy / dx) * 180 / np.pi
        elif dx == 0 and dy > 0:
            angle = 90
        elif dx < 0 <= dy:
            angle = 180 + np.arctan(dy / dx) * 180 / np.pi
        elif dx < 0 and dy < 0:
            angle = 180 + np.arctan(dy / dx) * 180 / np.pi
        elif dx == 0 and dy < 0:
            angle = 270
        elif dx > 0 >= dy:
            angle = 360 + np.arctan(dy / dx) * 180 / np.pi
        else:
            continue
        angles.append(angle)
    return angles


def _extract_edges(roi):
    """
    Extracts edges from the ROI by removing false edge pixels. Algorithm is
    explained in a flow-chart in the repo.

    .. figure:: ../_images/edges-before-dt.jpg
       :align: center

    .. figure:: ../_images/edges-with-noise.jpg
       :align: center

    .. figure:: ../_images/edges-after-dt.jpg
       :align: center

    :param numpy.ndarray roi: Region of Interest in image
    :return: Binary image containing edges
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    h, w, c = roi.shape
    ret, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(blur, ret / 3, ret)

    # The value of 200, could be changed, automatically, using geo information,
    # by taking into account the perimeter of the GCP in the image
    while cv2.countNonZero(edges) <= 200 and ret >= 150:
        ret = ret - 5
        edges = cv2.Canny(blur, ret / 3, ret)

    thresh1 = _rgb_threshold(roi, 180)
    thresh1 = morphology(thresh1, kernel)
    thresh2 = _drgb_threshold(roi)
    thresh2 = morphology(thresh2, kernel)
    thresh = cv2.bitwise_and(thresh1, thresh2)

    contours = extract_contours(thresh)
    contours = _check_contour_area(contours)
    contours = check_concavity(contours)
    bbox = _box_area(contours, roi)
    blank = np.zeros(edges.shape, np.uint8)
    if len(bbox) > 1:
        return blank

    idx = 0
    max_area = 0
    index = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            index = idx
            max_area = area
        idx = idx + 1
    points = []
    if index == -1:
        return np.zeros(edges.shape, np.uint8)

    for i in contours[index]:
        points.append(i[0])

    blank = np.zeros((h, w), np.uint8)
    cv2.fillPoly(blank, np.int32([points]), 255)
    blank = cv2.dilate(blank, kernel, 2)
    new_edges = cv2.bitwise_and(edges, blank)

    for i in range(h):
        for j in range(w):
            if edges[i][j]:
                dist = cv2.pointPolygonTest(contours[index], (j, i), True)
                if abs(dist) <= 5:
                    new_edges[i][j] = 255
    return new_edges


def _quiver_data(roi):
    """
    Calculates differential data for edge vector calculation, it outputs list
    of edge-pixels and corresponding derivatives

    :param numpy.ndarray roi: ROI in image
    :return: List of image-derivatives and position of edge-pixels i.e.
        dxs, dys, xs, ys
    """
    height, width, channels = roi.shape
    edges = _extract_edges(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    xs = []
    ys = []
    dxs = []
    dys = []
    for i in range(height):
        for j in range(width):
            if edges[i][j]:
                xs.append(j)
                ys.append(i)
                dxs.append(scharr_x[i][j])
                dys.append(scharr_y[i][j])
    return xs, ys, dxs, dys, edges


def detect_gcp(file_name):
    """
    This is the main function which uses other function from different modules
    to detect possible-gcps and output their probability in the image.

    :param str file_name: Path to image-file
    :return: List of co-ordinates of bounding-boxes and their probabilities
    """
    result = []

    # rotates jpeg
    rotate_jpeg(file_name)
    img = cv2.imread(file_name)

    # calculates image-intensity that could be used to set the threshold automatically
    # h, w, c = img.shape
    # avg = sum(np.ravel(img)) / (h * w * c)

    # Thresholds image in RGB-space and performs morphology on it
    thresh_rgb = _rgb_threshold(img, 180)
    thresh_rgb = morphology(thresh_rgb, kernel)

    # Thresholds image in DRGB-space and performs morphology on it
    thresh_Drgb = _drgb_threshold(img)
    thresh_Drgb = morphology(thresh_Drgb, kernel)

    # BITWISE and of the two binary images (RGB, DRGB)
    thresh = cv2.bitwise_and(thresh_rgb, thresh_Drgb)

    # Extracting valid contours after doing some checks, and drawing a bounding-box
    contours = extract_contours(thresh)
    contours1 = _check_contour_area(contours)
    contours2 = check_concavity(contours1)
    bboxs = _box_area(contours2, img)

    # Iterates over every bounding box and does verification at different stages
    rois = extract_roi(img, bboxs)
    for (roi, bbox) in zip(rois, bboxs):

        # Extracts differential data to calculate orientation
        xs, ys, dxs, dys, edges = _quiver_data(roi)
        orientations = _extract_angles(dxs, dys)

        # Generates a 9-bin histogram based on edge orientations after smoothening it
        bins = np.zeros((36, 1), np.uint)
        fig, ax = plt.subplots()
        data = ax.hist(orientations, 36, (0, 360))
        bins = np.transpose(bins)

        # Smoothing histogram-plot
        for i in range(36):
            if i == 0:
                bins[0][i] = data[0][i] + data[0][35] + data[0][i + 1]
            if i == 35:
                bins[0][i] = data[0][i - 1] + data[0][i] + data[0][0]
            else:
                bins[0][i] = data[0][i - 1] + data[0][i] + data[0][i + 1]

        # Peak Detection using impulses, that are almost 90-degree apart
        ans = np.zeros((1, 9), np.uint)
        for i in range(9):
            a = np.zeros((1, 36), np.uint)
            a[0][i] = 100
            a[0][9 + i] = 100
            a[0][18 + i] = 100
            a[0][27 + i] = 100
            ans[0][i] = sum(a[0] * bins[0])
        peak = int(np.argmax(ans[0]))
        stages = 0

        # Peak verification by looking at the local neighbourhood
        if (
            np.max(bins[0][max(peak - 2, 0) : peak + 2])
            >= np.max(bins[0][max(0, peak - 4) : peak + 4])
            >= 5
        ):
            stages = stages + 1

        if (
            np.max(bins[0][peak + 9 - 2 : peak + 9 + 2])
            >= np.max(bins[0][peak + 9 - 4 : peak + 9 + 4])
            >= 5
        ):
            stages = stages + 1

        if (
            np.max(bins[0][peak + 18 - 2 : peak + 18 + 2])
            >= np.max(bins[0][peak + 18 - 4 : peak + 18 + 4])
            >= 5
        ):
            stages = stages + 1

        if (
            np.max(bins[0][peak + 27 - 2 : min(peak + 27 + 2, 36)])
            >= np.max(bins[0][peak + 27 - 4 : min(peak + 27 + 4, 36)])
            >= 5
        ):
            stages = stages + 1

        # Probability of GCP-presence using trained-model
        edges = cv2.resize(edges, (28, 28))
        probability = classifier(edges)

        # Isometric weighting between the CV-algorithm and ML-model
        answer = 0.5 * stages / 4 + 0.5 * probability[0][0]

        # If the possible-GCP passes 3 or more stages in CV-algorithm
        # and has a probability of greater than  or equal to 0.75
        # and the total combined proability is greater than 0.875 than it is a GCP
        if stages / 4 >= 0.75 and probability[0][0] >= 0.75 and answer >= 0.875:
            result.append({"Co-ordinates": bbox, "Probability": answer})

        plt.close('all')

    return result
