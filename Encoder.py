import numpy as np
from ImageProcessor import *


class Encoder:
    """
    A class for encoding an image.

    Attributes
    ----------
    y_frame : ndarray or None
       The y frame of the image, if applicable.
    img : ndarray
       The input image to be encoded.
    qp : int
       Quantization parameter.
    """

    def __init__(self, img, qp=6, y_frame=None):
        """
        Initialize Encoder with the given image, quantization parameter, and y frame.

        Parameters
        ----------
        img : ndarray
            Input image for encoding.
        qp : int, optional
            Quantization parameter, by default 6.
        y_frame : ndarray, optional
            Y frame of the image, if applicable, by default None.
        """
        self.y_frame = y_frame
        self.img = np.asarray(img, dtype=np.uint8)
        self.qp = qp
        self.height, self.width = self.img.shape
        self.predicted_img = np.zeros(self.img.shape, dtype=np.uint8)
        self.intra_img = np.zeros(self.img.shape, dtype=np.uint8)
        self.recommended_modes = []
        self.image_processor = ImageProcessor(qp)
        self.offsets = {'A': (-1, 0), 'B': (-1, 1), 'C': (-1, 2), 'D': (-1, 3),
                        'E': (-1, 4), 'F': (-1, 5), 'G': (-1, 6), 'H': (-1, 7),
                        'I': (0, -1), 'J': (1, -1), 'K': (2, -1), 'L': (3, -1),
                        'Q': (-1, -1)}

    def fetch_predictions(self, i, j):
        """
        Fetch the predictions for the specified coordinates in the image.

        Parameters
        ----------
        i : int
            Row index.
        j : int
            Column index.

        Returns
        -------
        dict
            Dictionary of predictions.
        """
        predictions = {}
        for key, (di, dj) in self.offsets.items():
            new_i = np.clip(i + di, 0, self.height - 1)
            new_j = np.clip(j + dj, 0, self.width - 1)
            predictions[key] = self.predicted_img[new_i, new_j]
            if new_i != i + di or new_j != j + dj:
                predictions[key] = 128
        return predictions

    def full_image_prediction_for_compression(self):
        """
        Perform full image prediction for compression.

        Returns
        -------
        tuple
            A tuple containing the intra predicted image and the recommended modes.
        """
        for i in range(0, self.height, 4):
            for j in range(0, self.width, 4):
                predictions = self.fetch_predictions(i, j)
                if np.all(list(predictions.values()) == 128):
                    new_img_diff = np.zeros((4, 4))
                    new_img_temp = np.zeros((4, 4))
                    new_intra_temp = 128 * np.ones((4, 4))
                    new_img_diff = self.image_processor.compress_and_reconstruct_block(
                        self.img[i:i + 4, j:j + 4] - new_img_temp)
                    new_img_temp = new_img_diff + new_img_temp
                    self.predicted_img[i:i + 4, j:j + 4] = new_img_temp
                    self.intra_img[i:i + 4, j:j + 4] = new_intra_temp
                    self.recommended_modes.append(0)
                    continue
                sad_list = []
                for mode in range(0, 9):
                    block_recon_temp, new_intra_temp = self.image_processor.generate_prediction_block(
                        self.img[i:i + 4, j:j + 4], predictions, mode)
                    sad_list.append(DCTProcessor.SAD(block_recon_temp, self.img[i:i + 4, j:j + 4]))
                recommended_mode = np.argmin(sad_list)
                self.recommended_modes.append(recommended_mode)
                self.predicted_img[i:i + 4, j:j + 4], self.intra_img[i:i + 4,
                                                      j:j + 4] = self.image_processor.generate_prediction_block(
                    self.img[i:i + 4, j:j + 4], predictions, recommended_mode)
        self.intra_img = self.intra_img.astype(np.int32)
        return self.intra_img, self.recommended_modes

    def compress_and_reconstruct_full_img(self, diff_img):
        """
        Compress and reconstruct the full image.

        Parameters
        ----------
        diff_img : ndarray
            Difference image.

        Returns
        -------
        ndarray
            Reconstructed image.
        """
        return self.image_processor.compress_and_reconstruct_full_img(diff_img)

    def full_image_reconstruction(self, reconstructed_diff_img, modes):
        """
        Reconstruct the full image by adding the reconstructed difference image back to the predicted image.

        Parameters
        ----------
        reconstructed_diff_img : ndarray
            Reconstructed difference image.
        modes : list
            Modes for intra prediction.

        Returns
        -------
        ndarray
            The reconstructed image.
        """
        # Add the reconstructed difference image back to the predicted image
        reconstructed_img = self.predicted_img + reconstructed_diff_img
        return reconstructed_img

    def full_process(self, img):
        """
        Perform full process on the image, which includes prediction, compression, and reconstruction.

        Parameters
        ----------
        img : ndarray
            Input image.

        Returns
        -------
        tuple
            A tuple containing predicted image, difference image, reconstructed difference image, reconstructed image, and recommended modes.
        """
        predicted_img, modes = self.full_image_prediction_for_compression()
        diff_img = img - predicted_img
        reconstructed_diff_img = self.compress_and_reconstruct_full_img(diff_img)
        reconstructed_img = self.full_image_reconstruction(reconstructed_diff_img, modes)
        # reconstructed_img = np.asarray(reconstructed_img , dtype=np.int32)
        return predicted_img, diff_img, reconstructed_diff_img, reconstructed_img, modes
