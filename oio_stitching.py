from frame_orientation import rotate_image
import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm

# Frame slice denotes area which is used for OIO construction.
# In principle, we do not necessarily use the central part of the frame.
# We can use bottom or upper part of the frame as well.
# By this change, we can e.g. produce OIOs with different light conditions.
# If you want to change the slice used for OIO construction set this variable to 0.1
# (upper part) or 0.9 (bottom part)
FRAME_SLICE_START = 0.35

# We can blend the output from several frames. This can be useful especially in cases
# when data are noisy (i.e. strong radiation causes salt and pepper noise in images).
# In opposite, blending causes blurred output in cases when registration parameters
# were not properly estimated.
# 3 pixels is conservative blending. More is better for noisy frames, less is better
# in case of problematic registration.
AVERAGED_PIXELS = 3


def stitch_frame_sequence(frames, angles, y_shifts, x_shifts_cum):
    """
    Stitch a sequence of frames (images) into one composite image
    (also known as one image overview or OIO) based on the provided
    frame rotation angles and shifts in X and Y directions per frame.

    The frames are assumed to be captured by a moving camera with shifts and rotations
    for each captured frame specified in the same order. The final OIO will be of the size that
    fits all inputs frames, accounting their shifts and rotations.

    Args:
        frames (list of ndarray): A list of 2D numpy arrays with each array representing a grayscale image frame.
        angles (list of float): A list of rotation angles in degrees for each corresponding frame.
        y_shifts (list of float): A list of vertical shifts to be applied on each corresponding frame relative to the previous one.
        x_shifts_cum (list of int): A list of cumulative horizontal shifts to account for horizontal displacement over the frame sequence.

    Returns:
        oio (ndarray): The final stitched one image overview (OIO) as a 2D numpy array.
    """
    HEIGHT, WIDTH = frames[0].shape

    # Compute OIO width and height
    oio_height = int(np.sum(y_shifts))
    oio_width = int(np.max(x_shifts_cum) - np.min(x_shifts_cum) + WIDTH)

    # Create accumulator and weighting matrix
    accumulator = np.zeros((np.abs(oio_height), oio_width)).astype(float)
    weights = np.zeros((np.abs(oio_height), oio_width))

    # We start at the most top part of the OIO building the image.
    # NOTE: For videos going from bottom to the upper part more complex logic is necessary.
    position = 0

    for (idx, frame), angle in tqdm(zip(enumerate(frames), angles), total=len(frames), desc="Building OIO"):
        # Rotate the frame according to the angle
        rotated_frame = rotate_image(frame, angle)

        # Define frame subarea used for building OIO.
        slice_start = np.floor(HEIGHT * FRAME_SLICE_START).astype(int) - 1
        slice_height = np.abs(np.sum(
            y_shifts[idx: np.min([idx + AVERAGED_PIXELS, y_shifts.size])]
        ).astype(int))
        # Crop pixels which overflow the OIO height (we do not use them)
        if position + slice_height > np.abs(oio_height):
            slice_height = np.abs(oio_height) - position
        slice_end = np.ceil(slice_start + slice_height).astype(int) + 1

        # Do proper x alignment according to precomputed x_shifts
        row_px_position = np.arange(x_shifts_cum[idx], x_shifts_cum[idx] + WIDTH, 1)

        # Define position in the accumulator:
        acc_row_px_position = slice(0, WIDTH)
        acc_target = np.arange(np.floor(position).astype(int), np.ceil(position + slice_height).astype(int))
        extra_line = int(len(acc_target) - (slice_end - slice_start - 2))  # there is extra line (rounding issue)
        # Because y_shift is float number, some rows should be used only partially. For this reason we define
        # weight for each used frame row. Weights corresponds row usage ratio: e.g. for four rows weight
        # can be [0.2, 1, 1, 0.43]
        row_weight = np.array([1 - (position % 1)] +  # first row partial
                              [1 for _ in range(acc_target.size - 2)] +  # inner rows -> 1
                              [(position + slice_height) % 1]).reshape(-1, 1)  # last row partial

        # Build interpolation of the frame slice to exactly match int dimensions of accumulator
        interpolated = RectBivariateSpline(x=np.arange(slice_start, slice_end),
                                           y=np.arange(rotated_frame.shape[1]),
                                           z=rotated_frame[slice_start: slice_end, :],
                                           kx=np.min([slice_end - slice_start - 1, 3]))

        # Put slice into accumulator
        accumulator[acc_target, acc_row_px_position] += interpolated(np.arange(slice_start + 1, slice_end - 1 + extra_line), row_px_position) * row_weight
        # Increase weight accordingly
        weights[acc_target, acc_row_px_position] += np.ones((acc_target.size, row_px_position.size)) * row_weight

        # Move to the next slice
        position += y_shifts[idx]

    oio = accumulator
    oio[weights != 0] = accumulator[weights != 0] / weights[weights != 0]
    oio = ((oio - np.min(oio)) * 255 / (np.max(oio) - np.min(oio))).astype(np.uint8)

    return oio
