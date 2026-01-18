from scipy.stats import norm
from typing import Tuple
import numpy as np

def generate_grabbability_kernel(std_dev: int) -> np.ndarray:
    gk_1_by_4th = np.zeros(shape=(std_dev + 1, std_dev + 1))

    if std_dev == 0:
        gk_1_by_4th[0][0] = 1
        return gk_1_by_4th

    for i in range(std_dev + 1):
        for j in range(std_dev + 1):
            chessboard_distance = max(i, j)
            z_value = -chessboard_distance / std_dev
            gk_1_by_4th[i][j] = norm.cdf(z_value) * 2

    gk_bottom_half = np.concatenate(
        [np.flip(gk_1_by_4th[:, 1:], axis=1), gk_1_by_4th],
        axis=1
    )

    gk_full = np.concatenate(
        [np.flip(gk_bottom_half[1:, :], axis=0), gk_bottom_half],
        axis=0
    )

    return gk_full

def add_grabbability_kernel_to_image_inplace(image: np.ndarray, grabbability_kernel: np.ndarray, kernel_center_idx: Tuple[int, int]):
    gk_dim, _ = grabbability_kernel.shape

    # gk_dim = 2 * stddev + 1
    stddev = (gk_dim - 1) // 2

    d_rl = 0
    d_rh = 0
    d_cl = 0
    d_ch = 0

    if kernel_center_idx[0] - stddev < 0:
        d_rl = 0 - (kernel_center_idx[0] - stddev)
    if kernel_center_idx[0] + stddev > (image.shape[0] - 1):
        d_rh = (kernel_center_idx[0] + stddev) - (image.shape[0] - 1)

    if kernel_center_idx[1] - stddev < 0:
        d_cl = 0 - (kernel_center_idx[1] - stddev)
    if kernel_center_idx[1] + stddev > (image.shape[1] - 1):
        d_ch = (kernel_center_idx[1] + stddev) - (image.shape[1] - 1)
    
    image[
            max(0, kernel_center_idx[0] - stddev): min(image.shape[0], kernel_center_idx[0] + stddev + 1),
            max(0, kernel_center_idx[1] - stddev): min(image.shape[1], kernel_center_idx[1] + stddev + 1)
         ] += grabbability_kernel[d_rl: gk_dim - d_rh, d_cl: gk_dim - d_ch]
    
def subtract_grabbability_kernel_to_image_inplace(image: np.ndarray, grabbability_kernel: np.ndarray, kernel_center_idx: Tuple[int, int]):
    gk_dim, _ = grabbability_kernel.shape

    # gk_dim = 2 * stddev + 1
    stddev = (gk_dim - 1) // 2

    d_rl = 0
    d_rh = 0
    d_cl = 0
    d_ch = 0

    if kernel_center_idx[0] - stddev < 0:
        d_rl = 0 - (kernel_center_idx[0] - stddev)
    if kernel_center_idx[0] + stddev > (image.shape[0] - 1):
        d_rh = (kernel_center_idx[0] + stddev) - (image.shape[0] - 1)

    if kernel_center_idx[1] - stddev < 0:
        d_cl = 0 - (kernel_center_idx[1] - stddev)
    if kernel_center_idx[1] + stddev > (image.shape[1] - 1):
        d_ch = (kernel_center_idx[1] + stddev) - (image.shape[1] - 1)
    
    image[
            max(0, kernel_center_idx[0] - stddev): min(image.shape[0], kernel_center_idx[0] + stddev + 1),
            max(0, kernel_center_idx[1] - stddev): min(image.shape[1], kernel_center_idx[1] + stddev + 1)
         ] -= grabbability_kernel[d_rl: gk_dim - d_rh, d_cl: gk_dim - d_ch]