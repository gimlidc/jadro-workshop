import numpy as np
import cv2
import pandas as pd
from scipy.optimize import brute, minimize


def extract_grid_start_end_from_is_grid(is_grid, frame_no_processed_sequence_start=0):
    """
    Extracts the start and end frames of grids from a binary grid sequence.

    :param is_grid: list of binary values representing each frame in the sequence,
        where True indicates the presence of a grid
    :param frame_no_processed_sequence_start: (optional) starting frame number of the processed sequence

    :return: tuple containing the start and end frames of grids,
        and the average grid value for each grid
    """
    grids = []
    in_grid = False
    current_grid = []
    grids_start_end = []
    for frame_no, flag in enumerate(is_grid):
        if in_grid:
            if flag:
                current_grid.append(frame_no)
            else:
                in_grid = False
        else:
            if flag:
                # we are too close to previous grid
                if len(current_grid) > 0 and np.abs(current_grid[-1] - frame_no) < 50:
                    current_grid.append(frame_no)
                else:
                    if len(current_grid) > 0:
                        grids.append(np.nanmean(current_grid))
                        grids_start_end.append((current_grid[0] + frame_no_processed_sequence_start,
                                                current_grid[-1] + frame_no_processed_sequence_start))
                    current_grid = [frame_no]
                in_grid = True
    grids.append(np.nanmean(current_grid))
    grids_start_end.append(
        (current_grid[0] + frame_no_processed_sequence_start,
         current_grid[-1] + frame_no_processed_sequence_start)
    )

    # cleanup packs shorter than one frame
    grids_start_end = [grid_start_end for grid_start_end in grids_start_end if
                       np.abs(grid_start_end[1] - grid_start_end[0]) > 1]
    grids = [grid for grid_start_end, grid in zip(grids_start_end, grids) if
             np.abs(grid_start_end[1] - grid_start_end[0]) > 1],

    return grids_start_end, grids


def collect_video_grid_packs(images, grids_start_end=None):
    """
    According to grids start+ends collect frames and return them.
    This approach is 10 times faster than refuel.pipelines.grid_pack_extractor.collect_video_grid_packs
    """
    grids_images = []
    for start, end in grids_start_end:
        grids_images.append(images[start:end])
    return grids_images


def _generate_z_axis_from_speed_profile(speed_profile, rod_length_mm):
    # position of the frame in the OIO
    frame_position_px = np.cumsum(speed_profile)
    # we extend the OIO length with the speed of the last frame
    fa_length_px = frame_position_px[-1] + speed_profile[-1]

    # conversion speed_profile into milimeters for each frame
    frame_position_mm = [position_px / fa_length_px * rod_length_mm for position_px in frame_position_px]

    # export pixel_size_vertical
    pixels_per_mm_vertical = np.abs(fa_length_px / rod_length_mm)

    return pd.DataFrame(np.array([speed_profile,
                                  frame_position_px,
                                  frame_position_mm]).T,
                        columns=["shift px", "z-axis px", "z-axis mm"]), pixels_per_mm_vertical


def gauss_drop_outliers(vector, std_factor_distance):
    mean = np.nanmean(vector)
    std = np.nanstd(vector)
    outliers = (vector > (mean + std_factor_distance * std)) | (vector < (mean - std_factor_distance * std))
    out = vector.astype(float)
    out[outliers] = np.nan
    return out


def generate_speed_profile(
        sequence_frame_no_boundaries: tuple[int, int],
        grids_start_end: list[tuple[int, int]],
        speeds: list[tuple[float, float, list[float]]]
):
    frame_no_start, frame_no_end = sequence_frame_no_boundaries
    measured = np.array([measured_speed for s in speeds for measured_speed in s[2]])
    one_speed = np.nanmedian(gauss_drop_outliers(measured, 1))
    return np.ones(frame_no_end - frame_no_start) * one_speed


def estimate_shift(grid_pack):
    """
    Brute force approach to estimate the shift of each pair of consecutive frames in grid pack
    :param grid_pack: set of grayscale images of the spacer grid (horizontal edge must be present)
    :param tblr: necessary crop of the image
    :return: mean for all image pairs, standard deviation and measured nominals
    """
    sads = [np.sum(np.abs(np.diff(grid_img / 255.0, axis=1)), axis=1) for grid_img in grid_pack]
    sads_norm = np.array([sad / np.linalg.norm(sad) for sad in sads])
    shift_per_pair_of_consecutive_frames = []
    for sid in range(len(sads_norm) - 1):
        def conv(y_shift):
            xp = np.arange(len(sads_norm[sid + 1]))
            return np.sum(np.abs(
                sads_norm[sid] - np.roll(np.interp(xp + (y_shift % 1), xp, sads_norm[sid + 1]), int(y_shift // 1))))

        x0 = brute(conv, (slice(-15, 15, 1),), Ns=30)[0]
        shift_per_pair_of_consecutive_frames.extend(minimize(conv, [x0], bounds=[(x0 - 1, x0 + 1)]).x)
    return np.mean(shift_per_pair_of_consecutive_frames), np.std(
        shift_per_pair_of_consecutive_frames), shift_per_pair_of_consecutive_frames


def generate_z_axis_from_grids(is_grid, frames, rod_length_mm):
    grids_start_end, grids = extract_grid_start_end_from_is_grid(is_grid)
    grid_packs = collect_video_grid_packs(frames, grids_start_end)
    grid_speeds_px_per_frame = [estimate_shift(grid_pack) for grid_pack in grid_packs]

    speed_profile = generate_speed_profile(
        (0, len(frames)),
        grids_start_end,
        grid_speeds_px_per_frame,
    )

    return _generate_z_axis_from_speed_profile(speed_profile, rod_length_mm)
