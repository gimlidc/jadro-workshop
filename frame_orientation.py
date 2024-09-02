import numpy as np
import cv2


def rotate_image(image, rotation_angle_rad):
    """
    Simple function to rotate an image by a specified angle.
    """
    rotated_frame_target_shape = (image.shape[1], image.shape[0])
    degs = np.rad2deg(rotation_angle_rad)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), degs, 1)  # rotation is anticlockwise
    rotated_frame = cv2.warpAffine(image, rotation_matrix, rotated_frame_target_shape)

    return rotated_frame


def find_intersections(rho, theta, width, height):
    """ Find the intersections of the line with image borders to show HoughLines results"""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    # Points where the line intersects the image boundaries
    points = []

    # Intersection with the left boundary (x=0)
    if b != 0:
        y_left = int(rho / b)
        if 0 <= y_left <= height:
            points.append((0, y_left))

    # Intersection with the right boundary (x=width)
    if b != 0:
        y_right = int((rho - width * a) / b)
        if 0 <= y_right <= height:
            points.append((width, y_right))

    # Intersection with the top boundary (y=0)
    if a != 0:
        x_top = int(rho / a)
        if 0 <= x_top <= width:
            points.append((x_top, 0))

    # Intersection with the bottom boundary (y=height)
    if a != 0:
        x_bottom = int((rho - height * b) / a)
        if 0 <= x_bottom <= width:
            points.append((x_bottom, height))

    # Return the intersection points
    if len(points) == 2:
        return points[0], points[1]
    return None


def rotate_image_smaller_angle(img, angle_rad):
    """
    Rotate images on a half circle

    :param img: image
    :param angle_rad: rotation angle in radians (we rotate CCW)
    :return: darray
        Rotated image.
    """
    rows, cols = img.shape

    if angle_rad > np.pi / 2:
        angle_rad = angle_rad - np.pi

    degs = np.rad2deg(angle_rad)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degs, 1)  # rotation is anticlockwise
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst


def extract_undistorted_part(image):
    """
    Extracts central square that is not distorted
    :param image: input image
    :return: image central "undistorted" part
    """
    PREPROCESSING_ANGLE = np.pi / 4
    rows = image.shape[0]
    cols = image.shape[1]

    square_size = int(rows / 2)

    rows_to_skip = int((rows - square_size) / 2)
    cols_to_skip = int((cols - square_size) / 2)

    image = rotate_image_smaller_angle(image, PREPROCESSING_ANGLE)
    undistorted = image[rows_to_skip:rows_to_skip + square_size,
                  cols_to_skip:cols_to_skip + square_size]

    return undistorted, np.rad2deg(PREPROCESSING_ANGLE)