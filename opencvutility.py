"""
.. versionadded:: 0.2
.. codeauthor:: Aman Gajendra Jain <aj161198@gmail.com>

**External Dependencies**

.. hlist::
   :columns: 3

   - cv2
   - piexif
   - PIL

**Internal Dependencies**

.. hlist::
   :columns: 1

   - :class:`~quark.helpers.spatial`
"""

# Copyright (C) 2018 Skylark Drones

import cv2
import piexif
from PIL import Image

from spatial import is_point_inside_polygon


def rotate_jpeg(filename):
    """
    Rotates and overwrites the image

    :param str filename: Absolute path to the image-file
    """
    image = Image.open(filename)

    if 'exif' not in image.info:
        return

    exif_dict = piexif.load(image.info["exif"])

    if piexif.ImageIFD.Orientation not in exif_dict["0th"]:
        return

    orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
    exif_bytes = piexif.dump(exif_dict)

    if orientation == 2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        image = image.rotate(180)
    elif orientation == 4:
        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 5:
        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 7:
        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        image = image.rotate(90, expand=True)

    # TODO: Write the new orientation of the image into the EXIF metadata

    image.save(filename, exif=exif_bytes)


def rect_contains(rect, pt):
    """
    Checks whether a given point is inside the rectangle

    :param list rect: The bounding-box to be checked for a point in it
    :param tuple pt: The point to be checked
    :return: if the point is inside the bounding-box or not
    :rtype: bool
    """
    polygon = [
        (rect[0], rect[1]),
        (rect[0], rect[1] + rect[3]),
        (rect[0] + rect[2], rect[1] + rect[3]),
        (rect[0] + rect[2], rect[1]),
    ]
    return is_point_inside_polygon(pt, polygon)


def morphology(black_white, kernel):
    """
    Performs the preliminary morphological operations to fill the holes and
    remove noise i.e, it performs morphological opening and closing operations

    :param numpy.ndarray black_white: The binary image on which the operation
        is to be performed
    :param numpy.ndarray kernel: The type of structuring-element to be used for
        performing this operations
    :return: Binary image, on which morphological operations were performed
    """
    closing = cv2.morphologyEx(black_white, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return opening


def extract_contours(black_white):
    """
    Extract only "External" contours from the binary image

    :param numpy.ndarray black_white: The binary image from which the contours
        were to be extracted
    :return: External contours
    :rtype: list
    """
    black_white, contours, h = cv2.findContours(
        black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def check_concavity(contours):
    """
    Checks for concavity for a list of contours, and appends it to the new list
    if they are concave.
    
    .. note::
        The argument that specifies approximation, i.e.
        "0.01 * cv2.arcLength(contour, True)" could be changed based on the
        requirement.

    :param list contours: Contours to be checked for concavity
    :return: Contours that are concave
    :rtype: list
    """
    concave = []
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True
        )
        if ~cv2.isContourConvex(approx):
            concave.append(contour)

    return concave


def extract_roi(image, bbox):
    """
    Extracts a list of region-of-interest(ROI) from the list of bounding-boxes
    from a given image

    :param numpy.ndarray image: The image from which ROI's were to be extracted
    :param list bbox: The list of bounding-boxes, where each bounding box is
        given by [x,y,w,h] i.e [x-coordinate of top-left corner, y-coordinate
        of top-left corner, width of rectangle, height of rectangle]
    :return: ROIs extracted from the image
    :rtype: list
    """
    rois = []
    for x, y, w, h in bbox:
        roi = image[y : y + h - 1, x : x + h - 1]
        rois.append(roi)
    return rois
