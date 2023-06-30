import numpy as np


class DCTProcessor:
    """
    A class to process Discrete Cosine Transform (DCT) operations on image blocks.

    Attributes
    ----------
    qp : int
        Quantization parameter.
    mr_values : ndarray
        MR values used in quantization and dequantization matrices.
    vr_values : ndarray
        VR values used in quantization and dequantization matrices.
    qp_mod_6 : int
        Remainder of the quantization parameter divided by 6.
    """
    def __init__(self, qp):
        """
        Initialize DCTProcessor with the given quantization parameter (qp).

        Parameters
        ----------
        qp : int
            Quantization parameter.
        """
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
        """
        Perform 4x4 Discrete Cosine Transform (DCT) on a given block.

        Parameters
        ----------
        block : ndarray
            Input block for DCT.

        Returns
        -------
        ndarray
            Block after DCT.
        """
        dct_mat = np.array([
            [1, 1, 1, 1],
            [2, 1, -1, -2],
            [1, -1, -1, 1],
            [1, -2, 2, -1]
        ], dtype=np.int32)
        return dct_mat @ block @ dct_mat.T

    def _create_matrix(self, arr):
        """
        Create a matrix from the given array using the quantization parameter.

        Parameters
        ----------
        arr : ndarray
            Input array.

        Returns
        -------
        ndarray
            Created matrix.
        """
        idx = self.qp_mod_6
        M = np.array([
            [arr[0][idx], arr[2][idx], arr[0][idx], arr[2][idx]],
            [arr[2][idx], arr[1][idx], arr[2][idx], arr[1][idx]],
            [arr[0][idx], arr[2][idx], arr[0][idx], arr[2][idx]],
            [arr[2][idx], arr[1][idx], arr[2][idx], arr[1][idx]]
        ], dtype=np.int32)
        return M

    def get_quantization_matrix(self):
        """
        Get the quantization matrix.

        Returns
        -------
        ndarray
            Quantization matrix.
        """
        return self._create_matrix(self.mr_values)

    def get_dequantization_matrix(self):
        """
        Get the dequantization matrix.

        Returns
        -------
        ndarray
            Dequantization matrix.
        """
        return self._create_matrix(self.vr_values)

    def quantization(self, quant_coefficients):
        """
        Perform quantization on the given coefficients.

        Parameters
        ----------
        quant_coefficients : ndarray
            Coefficients for quantization.

        Returns
        -------
        ndarray
            Quantized coefficients.
        """
        return np.round(quant_coefficients * self.get_quantization_matrix() * (1 / (2 ** (15 + self.qp / 6))))

    def dequantization(self, quant_coefficients):
        """
        Perform dequantization on the given coefficients.

        Parameters
        ----------
        quant_coefficients : ndarray
            Coefficients for dequantization.

        Returns
        -------
        ndarray
            Dequantized coefficients.
        """
        return quant_coefficients * self.get_dequantization_matrix() * np.floor(2 ** (self.qp / 6))

    def idct_4x4(self, block):
        """
        Perform 4x4 Inverse Discrete Cosine Transform (IDCT) on a given block.

        Parameters
        ----------
        block : ndarray
            Input block for IDCT.

        Returns
        -------
        ndarray
            Block after IDCT.
        """
        idct_mat = np.array([
            [1, 1, 1, 0.5],
            [1, 0.5, -1, -1],
            [1, -0.5, -1, 1],
            [1, -1, 1, -0.5]
        ])
        return (idct_mat @ block @ idct_mat.T) * (1 / (2 ** 6))
    @staticmethod
    def calculate_psnr(img1, img2):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) for two images.

        Parameters
        ----------
        img1, img2 : ndarray
            Two images to compare.

        Returns
        -------
        float
            PSNR of the two images.
        """
        mse = np.mean((img1 - img2) ** 2)
        max_pixel_value = np.max(img1)
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_mad(block1, block2):
        """
        Calculate Mean Absolute Difference (MAD) for two blocks.

        Parameters
        ----------
        block1, block2 : ndarray
            Two blocks to compare.

        Returns
        -------
        float
            MAD of the two blocks.
        """
        return np.mean(np.abs(block1 - block2))

    @staticmethod
    def SAD(block1, block2):
        """
        Calculate Sum of Absolute Differences (SAD) for two blocks.

        Parameters
        ----------
        block1, block2 : ndarray
            Two blocks to compare.

        Returns
        -------
        float
            SAD of the two blocks.
        """
        return np.sum(np.abs(block1 - block2))