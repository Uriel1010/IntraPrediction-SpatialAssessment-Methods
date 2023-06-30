import numpy as np


class DCTProcessor:

    def __init__(self, qp):
        self.qp = qp
        self.mr_values = np.array([
            [13107, 11916, 10082, 9362, 8192, 7282],
            [5243, 4660, 4194, 3647, 3355, 2893],
            [8066, 7490, 6554, 5825, 5243, 4559]
        ], dtype=np.int32)

        self.vr_values = np.array([
            [10, 11, 13, 14, 16, 18],
            [16, 18, 20, 23, 25, 29],
            [13, 14, 16, 18, 20, 23]
        ], dtype=np.int32)

        self.qp_mod_6 = self.qp % 6

    def dct_4x4(self, block):
        dct_mat = np.array([
            [1, 1, 1, 1],
            [2, 1, -1, -2],
            [1, -1, -1, 1],
            [1, -2, 2, -1]
        ], dtype=np.int32)
        return dct_mat @ block @ dct_mat.T

    def _create_matrix(self, arr):
        idx = self.qp_mod_6
        M = np.array([
            [arr[0][idx], arr[2][idx], arr[0][idx], arr[2][idx]],
            [arr[2][idx], arr[1][idx], arr[2][idx], arr[1][idx]],
            [arr[0][idx], arr[2][idx], arr[0][idx], arr[2][idx]],
            [arr[2][idx], arr[1][idx], arr[2][idx], arr[1][idx]]
        ], dtype=np.int32)
        return M

    def get_quantization_matrix(self):
        return self._create_matrix(self.mr_values)

    def get_dequantization_matrix(self):
        return self._create_matrix(self.vr_values)

    def quantization(self, quant_coefficients):
        return np.round(quant_coefficients * self.get_quantization_matrix() * (1 / (2 ** (15 + self.qp / 6))))

    def dequantization(self, quant_coefficients):
        return quant_coefficients * self.get_dequantization_matrix() * np.floor(2 ** (self.qp / 6))

    def idct_4x4(self, block):
        idct_mat = np.array([
            [1, 1, 1, 0.5],
            [1, 0.5, -1, -1],
            [1, -0.5, -1, 1],
            [1, -1, 1, -0.5]
        ])
        return (idct_mat @ block @ idct_mat.T) * (1 / (2 ** 6))
    @staticmethod
    def calculate_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        max_pixel_value = np.max(img1)
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_mad(block1, block2):
        return np.mean(np.abs(block1 - block2))

    @staticmethod
    def SAD(block1, block2):
        return np.sum(np.abs(block1 - block2))