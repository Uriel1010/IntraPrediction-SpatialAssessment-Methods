import numpy as np

from main import intra_prediction


class Decoder:
    """
    A class for decoding a given image with Intra Prediction.

    Attributes
    ----------
    img : ndarray
        The input image to be decoded.
    modes : list
        The list of modes for Intra Prediction.
    height : int
        Height of the image.
    width : int
        Width of the image.
    reconstructed_img : ndarray
        The reconstructed image after decoding.
    offsets : dict
        Dictionary of offsets for fetching predictions.
    """

    def __init__(self, img, modes):
        """
        Initialize Decoder with the given image and modes.

        Parameters
        ----------
        img : ndarray
            Input image for decoding.
        modes : list
            Modes for Intra Prediction.
        """
        self.img = np.asarray(img, dtype=np.int32)
        self.modes = modes
        self.height, self.width = self.img.shape
        self.reconstructed_img = np.zeros(self.img.shape, dtype=np.int32)
        self.offsets = {'A': (-1, 0), 'B': (-1, 1), 'C': (-1, 2), 'D': (-1, 3), 'E': (-1, 4),
                        'F': (-1, 5), 'G': (-1, 6), 'H': (-1, 7), 'I': (0, -1), 'J': (1, -1),
                        'K': (2, -1), 'L': (3, -1), 'Q': (-1, -1)}

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
            predictions[key] = self.reconstructed_img[new_i, new_j]
            if new_i != i + di or new_j != j + dj:
                predictions[key] = 128
        return predictions

    def full_image_reconstruction(self):
        """
        Reconstruct the full image by applying the decoding process block by block.

        Returns
        -------
        ndarray
            The reconstructed image.
        """
        block_index = 0
        for i in range(0, self.height, 4):
            for j in range(0, self.width, 4):
                mode = self.modes[block_index]
                block_index += 1
                predictions = self.fetch_predictions(i, j)
                if np.all(list(predictions.values()) == 128):
                    self.reconstructed_img[i:i + 4, j:j + 4] = 128 + self.img[i:i + 4, j:j + 4]
                    continue
                intra_predicted_block = intra_prediction(predictions, mode)
                self.reconstructed_img[i:i + 4, j:j + 4] = intra_predicted_block + self.img[i:i + 4, j:j + 4]
        return self.reconstructed_img
