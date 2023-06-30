import numpy as np
from ImageProcessor import *

class Encoder:
    def __init__(self, img, qp=6, y_frame=None):
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
        predictions = {}
        for key, (di, dj) in self.offsets.items():
            new_i = np.clip(i + di, 0, self.height - 1)
            new_j = np.clip(j + dj, 0, self.width - 1)
            predictions[key] = self.predicted_img[new_i, new_j]
            if new_i != i + di or new_j != j + dj:
                predictions[key] = 128
        return predictions

    def full_image_prediction_for_compression(self):
        for i in range(0, self.height, 4):
            for j in range(0, self.width, 4):
                predictions = self.fetch_predictions(i, j)
                if np.all(list(predictions.values()) == 128):
                    new_img_diff = np.zeros((4,4))
                    new_img_temp = np.zeros((4,4))
                    new_intra_temp = 128*np.ones((4,4))
                    new_img_diff = self.image_processor.compress_and_reconstruct_block(self.img[i:i + 4, j:j + 4] - new_img_temp)
                    new_img_temp = new_img_diff + new_img_temp
                    self.predicted_img[i:i+4, j:j+4] = new_img_temp
                    self.intra_img[i:i+4, j:j+4] = new_intra_temp
                    self.recommended_modes.append(0)
                    continue
                sad_list = []
                for mode in range(0, 9):
                    block_recon_temp, new_intra_temp = self.image_processor.generate_prediction_block(self.img[i:i + 4, j:j + 4], predictions, mode)
                    sad_list.append(DCTProcessor.SAD(block_recon_temp, self.img[i:i + 4, j:j + 4]))
                recommended_mode = np.argmin(sad_list)
                self.recommended_modes.append(recommended_mode)
                self.predicted_img[i:i+4, j:j+4], self.intra_img[i:i+4, j:j+4] = self.image_processor.generate_prediction_block(self.img[i:i+4, j:j+4], predictions, recommended_mode)
        self.intra_img = self.intra_img.astype(np.int32)
        return self.intra_img, self.recommended_modes

    def compress_and_reconstruct_full_img(self, diff_img):
        return self.image_processor.compress_and_reconstruct_full_img(diff_img)

    def full_image_reconstruction(self, reconstructed_diff_img, modes):
        # Add the reconstructed difference image back to the predicted image
        reconstructed_img = self.predicted_img + reconstructed_diff_img
        return reconstructed_img

    def full_process(self, img):
        predicted_img, modes = self.full_image_prediction_for_compression()
        diff_img = img - predicted_img
        reconstructed_diff_img = self.compress_and_reconstruct_full_img(diff_img)
        reconstructed_img = self.full_image_reconstruction(reconstructed_diff_img, modes)
        # reconstructed_img = np.asarray(reconstructed_img , dtype=np.int32)
        return predicted_img, diff_img, reconstructed_diff_img, reconstructed_img, modes

