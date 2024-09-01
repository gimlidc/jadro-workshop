import numpy as np
from tqdm.auto import tqdm
from scipy import optimize
import cv2


def shifted_triplet_slice(frame, overlap, abs_int_y_shift):
    return np.concatenate([
        frame[abs_int_y_shift: abs_int_y_shift + overlap],
        frame[frame.shape[0] // 2 + abs_int_y_shift: frame.shape[0] // 2 + abs_int_y_shift + overlap],
        frame[-overlap:]
    ], axis=0)


def base_triplet_slice(frame, overlap, abs_int_y_shift):
    return np.concatenate([
        frame[:overlap],
        frame[frame.shape[0] // 2: frame.shape[0] // 2 + overlap],
        frame[-overlap - abs_int_y_shift: -abs_int_y_shift]
    ], axis=0)


def __get_img_bands(frame, ref_frame, int_shift_y, overlap_height=60):
    """
    A function returning only registered parts of two consecutive images;
    the selected bands from entire images are used to calculate convolution

    :param frame:
        Registered part of image 1
    :param ref_frame:
        Registered part of the following image 2
    :param int_shift_y:
        Integered and rounded current best shift in y that is needed to determine the parts of images
        that show the same physical area of the fuel set
    :param overlap_height: int
        Number of pixels the two images will share = height of both image bands returned here

    :return img1_band: ndarray
        A selected part of image 1 showing the same physical area as img2_band
    :return img2_band: ndarray
        A selected part of image 2 showing the same physical part as img1_band
    """
    slice_height_third = overlap_height // 3  # taking 1/3 of entire overlap height

    return (shifted_triplet_slice(frame, slice_height_third, int_shift_y),
            base_triplet_slice(ref_frame, slice_height_third, int_shift_y))

    # for upward movement
    #return (base_triplet_slice(frame, slice_height_third, int_shift_y),
    #        shifted_triplet_slice(ref_frame, slice_height_third, int_shift_y))


def safe_grayscale_double(img):
    if img.shape is None:
        return None
    if len(img.shape) == 3:
        return np.float64(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        return np.float64(img)


def norm_row(row):
    """ Normalization of row intensities.
    After processing mean will be 0 and variance 1. """

    if np.var(row) != 0:
        return (row - np.mean(row)) / np.std(row)
    else:
        return row - np.mean(row)


def __norm_rows_with_preprocess(frame_a, frame_b):
    """
    Normalize some image bands after optional cropping
    (if parameter overlap_height is not None) and converting
    to np.float64 grayscale.

    :param frame_a: darray
        First frame, or selected image band or even a single row
    :param frame_b: darray
        Second frame to process
    :return: row_a, row_b
        Normalized, grayscaled and cropped image bands (or rows)
    """

    row_a = norm_row(safe_grayscale_double(frame_a))
    row_b = norm_row(safe_grayscale_double(frame_b))

    return row_a, row_b


def __row_convolution(row_a, row_b):
    """
    Apply convolution on two matrices (1D or 2D)

    :param row_a: darray
        matrix A
    :param row_b: darray
        matrix B
    :return:
        ΣΣ(A.*B)
    """

    if len(row_a.shape) == 1:
        return sum(np.multiply(np.float64(row_a), np.float64(row_b)))
    else:
        return sum(sum(np.multiply(np.float64(row_a), np.float64(row_b))))


def __row_shift_fitness(row_a, row_b, x_shift):
    """
    Method shift rows with x_shift and return convolution of them

    :param x_shift: double
        shift along the x-axis
    :param row_a:
        matrix A
    :param row_b:
        matrix B
    :return: double
        scalar - the result of convolution of two shifted rows
    """

    rows, cols = row_a.shape
    # NOTE: a half shift is used to persist same vector length for convolution
    row_a_shifted = cv2.warpAffine(
        row_a,
        np.float32([[1, 0, -x_shift/2], [0, 1, 0]]),
        ((cols + np.abs(x_shift)/2).astype(int), rows),
        flags=cv2.INTER_CUBIC
    ),
    row_b_shifted = cv2.warpAffine(
        row_b,
        np.float32([[1, 0, x_shift/2], [0, 1, 0]]),
        ((cols + np.abs(x_shift)/2).astype(int), rows),
        flags=cv2.INTER_CUBIC
    ),

    return __row_convolution(row_a_shifted[0], row_b_shifted[0])


def compute_best_x_translation_real(frame_a, frame_b, max_abs_shift):
    """
    Estimator for xShift in real numbers
    :param frame_a: image
        Fixed input
    :param frame_b: image
        Moving input
    :param max_abs_shift: int
        number bounding interval in which method search for best solution
    :return: float
        Approximate xShift for frame_b to match frame_a position
    """

    # normalize rows after safe grayscale transformation and eventual cropping
    row_a, row_b = __norm_rows_with_preprocess(frame_a, frame_b)

    def fun(x_shift):
        return -__row_shift_fitness(row_a, row_b, x_shift)

    out = optimize.minimize_scalar(fun, bounds=(-max_abs_shift, max_abs_shift), method='bounded')
    # NOTE: this can be used for higher random walk:
    # out = optimize.differential_evolution(fun, bounds=((-max_abs_shift, 0), (max_abs_shift, 0)),
    # popsize=100, mutation=1, disp=True)
    # print(out)
    return out.x


def __fill_1d_array_with_zeroes_left_right(vector, amount):

    if amount == 0:
        return vector

    out = np.zeros(vector.shape[0] + np.abs(amount))
    if amount < 0:
        out[-amount:] = vector
    else:
        out[:-amount] = vector

    return out


def __fill_2d_array_with_zeroes_left_right(matrix, amount):

    if amount == 0:
        return matrix

    out = np.zeros([matrix.shape[0], matrix.shape[1] + np.abs(amount)])
    if amount < 0:
        out[:, :amount] = matrix
    else:
        out[:, amount:] = matrix

    return out


def row_convolution_with_int_shift(row_a, row_b, maxAbsShift):
    """
    Method computes convolution values for two image rows which one will be shifted. Shift is computed only for positive
    integers.

    The second row is shifted by [-maxAbsShift, +maxAbsShift]. The frist row is fixed. Produced is an array of length
    2 * maxAbsShift + 1 with convolution value

    :param row_a: intensity values
        Gray scale intensity values of image row. This row is fixed.
    :param row_b:
        Gray scale intensity values of image row. This row is moving.
    :return:
        Array of size 2 * maxAbsShift + 1 where all convolution results are computed for every possible shift amount
    """
    out = np.float64(np.zeros(2 * maxAbsShift + 1))
    for xShift in range(-maxAbsShift, maxAbsShift + 1):

        try:
            if len(row_a.shape) == 1:
                row_a_filled_zeroes = __fill_1d_array_with_zeroes_left_right(row_a, -xShift)
                row_b_filled_zeroes = __fill_1d_array_with_zeroes_left_right(row_b, xShift)
                out[maxAbsShift + xShift] = __row_convolution(row_a_filled_zeroes, row_b_filled_zeroes) / (len(row_a) - xShift)
            else:
                row_a_filled_zeroes = __fill_2d_array_with_zeroes_left_right(row_a, -xShift)
                row_b_filled_zeroes = __fill_2d_array_with_zeroes_left_right(row_b, xShift)

                # normalization to the number of pixels is meant to avoid auto-regularization
                normalization = row_a.shape[0] * (row_a.shape[1] - xShift)
                out[maxAbsShift + xShift] = __row_convolution(row_a_filled_zeroes, row_b_filled_zeroes) / normalization
        except Exception as a:
            print(row_a.shape, row_b.shape, xShift)
            raise a

    return out


def compute_best_x_translation(frame_a, frame_b):
    """
    Method computes x-axis shift for two grayscale images with vertically oriented fuel sets
    :param frame_a: image
        The upper part of a fuel set
    :param frame_b:
        The bottom part of a fuel set
    :return:
        Shift in pixels along the x-axis (to the left are negative values)
    """

    frame_a_normed_rows = norm_row(frame_a)
    frame_b_normed_rows = norm_row(frame_b)

    max_abs_shift = 5
    convolution_array = row_convolution_with_int_shift(frame_a_normed_rows,
                                                       frame_b_normed_rows,
                                                       max_abs_shift)

    position = np.argmax(convolution_array)

    return position - max_abs_shift


def optimize_frame_shift(frame_slice, ref_register):
    if ref_register is not None:
        best_shift_x_int = compute_best_x_translation(ref_register,
                                                      frame_slice)

        slice_shifted = np.roll(frame_slice, best_shift_x_int)  # shift the img2_band by integer shift found
        # now for the already int-shifted img use small max_abs_shift_x (1,2) to find best x-shift with finer resolution
        best_shift_x_real_from_int = compute_best_x_translation_real(ref_register,
                                                                     slice_shifted,
                                                                     max_abs_shift=1)  # 1 should be enough (theoretically)
        current_best_shift_x = best_shift_x_int + best_shift_x_real_from_int
        return current_best_shift_x
    else:
        return 0


def estimate_x_shifts(frames, y_shifts):
    prev_frame = frames[0]
    shifts = [0]
    for current_frame, shift in tqdm(zip(frames[1:], y_shifts),
                                     desc=f"Shift estimation",
                                     total=len(frames) - 1):
        frame_slice, ref_register = __get_img_bands(current_frame, prev_frame, int_shift_y=int(np.abs(shift)))
        shifts.append(optimize_frame_shift(frame_slice, ref_register))
        prev_frame = current_frame
    return np.array(shifts).astype(float)