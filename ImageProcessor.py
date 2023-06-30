from DCTProcessor import *

class ImageProcessor:
    """
    A class for processing an image using Discrete Cosine Transform (DCT).

    Attributes
    ----------
    qp : int
        Quantization parameter.
    dct_processor : DCTProcessor
        Instance of DCTProcessor.
    """
    def __init__(self, qp):
        """
        Initialize ImageProcessor with the given quantization parameter (qp).

        Parameters
        ----------
        qp : int
            Quantization parameter.
        """
        self.qp = qp
        self.dct_processor = DCTProcessor(qp)

    def compress_and_reconstruct_block(self, block):
        """
        Compress and reconstruct a block using DCT.

        Parameters
        ----------
        block : ndarray
            Input block for compression and reconstruction.

        Returns
        -------
        ndarray
            Reconstructed block.
        """
        dct_block = self.dct_processor.dct_4x4(block)
        quantized_block = self.dct_processor.quantization(dct_block)
        dequantized_block = self.dct_processor.dequantization(quantized_block)
        reconstructed_block = self.dct_processor.idct_4x4(dequantized_block)
        return reconstructed_block

    def compress_and_reconstruct_full_img(self, intra_predicted):
        """
        Compress and reconstruct the full image using DCT.

        Parameters
        ----------
        intra_predicted : ndarray
            Intra predicted image for compression and reconstruction.

        Returns
        -------
        ndarray
            Reconstructed image.
        """
        reconstructed_img = np.zeros(intra_predicted.shape, dtype=np.int32)
        for i in range(0, intra_predicted.shape[0], 4):
            for j in range(0 , intra_predicted.shape[1] , 4):
                reconstructed_img[i:i+4, j:j+4] = self.compress_and_reconstruct_block(intra_predicted[i:i+4, j:j+4])
        return np.asarray(reconstructed_img, dtype=np.int32)

    def generate_prediction_block(self, original_block, predictions, mode):
        """
        Generate a prediction block using Intra Prediction.

        Parameters
        ----------
        original_block : ndarray
            Original block for prediction.
        predictions : dict
            Predictions for generating a new block.
        mode : int
            Intra prediction mode.

        Returns
        -------
        tuple
            A tuple containing new block and new Intra predicted block.
        """
        A, B, C, D, E, F, G, H, I, J, K, L, Q = predictions.values()
        new_intra = intra_prediction(original_block, mode, A, B, C, D, E, F, G, H, I, J, K, L, Q)
        new_block_diff = self.compress_and_reconstruct_block(original_block - new_intra)
        new_block = new_block_diff + new_intra
        return new_block, new_intra


def intra_prediction(block, mode, A, B, C, D, E, F, G, H, I, J, K, L, Q):
    """
    Performs intra prediction on a 4x4 block.

    Parameters
    ----------
    block : ndarray
        A 4x4 block of pixels.
    mode : int
        The intra prediction mode (0-8).
    A, B, C, D, E, F, G, H, I, J, K, L, Q : int
        Pixel values from surrounding blocks used for prediction.

    Returns
    -------
    intra_predicted_block : ndarray
        A 4x4 block of predicted pixels.
    """
    if not 0 <= mode <= 8:
        raise ValueError("Mode must be an integer between 0 and 8.")

    intra_predicted_block = np.zeros((4, 4))

    if mode == 2:  # DC prediction
        P = (sum([A, B, C, D, I, J, K, L]) + 4) // 8
        intra_predicted_block = np.full((4, 4), P)
    elif mode == 0:  # Vertical prediction
        intra_predicted_block = np.array([[A, B, C, D] for _ in range(4)])
    elif mode == 1:  # Horizontal prediction
        intra_predicted_block = np.array([[I, J, K, L] for _ in range(4)])
    elif mode == 3:  # Diagonal down-left prediction
        vals = [(A + 2 * B + C + 2) / 4, (B + 2 * C + D + 2) / 4, (C + 2 * D + E + 2) / 4,
                (D + 2 * E + F + 2) / 4, (E + 2 * F + G + 2) / 4, (F + 2 * G + H + 2) / 4,
                (G + 3 * H + 2) / 4]
        arrange = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        for numR, row in enumerate(arrange):
            for numC, column in enumerate(row):
                intra_predicted_block[numR][numC] = vals[column]
    elif mode == 4:  # Diagonal down-right prediction
        vals = [(L + 2 * K + J + 2) / 4, (K + 2 * J + I + 2) / 4, (J + 2 * I + Q + 2) / 4,
                (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4,
                (B + 2 * C + D + 2) / 4]
        arrange = [[3, 4, 5, 6], [2, 3, 4, 5], [1, 2, 3, 4], [0, 1, 2, 3]]
        for numR, row in enumerate(arrange):
            for numC, column in enumerate(row):
                intra_predicted_block[numR][numC] = vals[column]
        # return np.diag(block)
    elif mode == 5:
        # Vertical right prediction
        intra_predicted_block = np.array([[(Q + A + 1) / 2, (A + B + 1) / 2, (B + C + 1) / 2, (C + D + 1) / 2],
                                         [(I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4,
                                          (B + 2 * C + D + 2) / 4],
                                         [(Q + 2 * I + J + 2) / 4, (Q + A + 1) / 2, (A + B + 1) / 2, (B + C + 1) / 2],
                                         [(I + 3 * J + K + 2) / 4, (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4,
                                          (A + 2 * B + C + 2) / 4]])
        # return np.roll(block, -1, axis=1)
    elif mode == 6:
        # Horizontal down prediction
        intra_predicted_block = np.array(
            [[(Q + I + 1) / 2, (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4],
            [(I + J + 1) / 2, (Q + 2 * I + J + 2) / 4, (Q + I + 1) / 2, (I + 2 * Q + A + 2) / 4],
            [(J + K + 1) / 2, (I + 2 * J + K + 2) / 4, (I + J + 1) / 2, (Q + 2 * I + J + 2) / 4],
            [(K + L + 1) / 2, (J + 2 * K + L + 2) / 4, (J + K + 1) / 2, (I + 2 * J + K + 2) / 4]])
        # return np.roll(block, -1, axis=0)
    elif mode == 7:
        # Vertical left prediction
        intra_predicted_block = np.array([[(A + B + 1) / 2, (B + C + 1) / 2, (C + D + 1) / 2, (D + E + 1) / 2],
                                         [(A + 2 * B + C + 2) / 4, (B + 2 * C + D + 2) / 4, (C + 2 * D + E + 2) / 4,
                                          (D + 2 * E + F + 2) / 4],
                                         [(B + C + 1) / 2, (C + D + 1) / 2, (D + E + 1) / 2, (E + F + 1) / 2],
                                         [(C + D + 1) / 2, (D + E + 1) / 2, (E + F + 1) / 2, (E + 2 * F + G + 2) / 4]])
        # return np.roll(block, 1, axis=1)
    elif mode == 8:
        # Horizontal up prediction
        intra_predicted_block = np.array(
            [[(I + J + 1) / 2, (I + 2 * J + K + 2) / 4, (J + K + 1) / 2, (J + 2 * K + L + 2) / 4],
            [(J + K + 1) / 2, (J + 2 * K + L + 2) / 4, (K + L + 1) / 2, (K + 2 * L + L + 2) / 4],
            [(K + L + 1) / 2, (K + 2 * L + L + 2) / 4, L, L],
            [L, L, L, L]])
        # return np.roll(block, 1, axis=0)
    return intra_predicted_block
