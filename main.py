import os

import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import pprint
from DCTProcessor import *
from Encoder import *
from PIL import Image


def save_plot(plot, filename):
    save_path = os.path.join('files', 'output_files', filename)
    plt.savefig(save_path)


def count_modes(modes):
    count = np.zeros(9, dtype=int)
    for idx in range(len(modes)):
        if idx == 0:
            continue
        if modes[idx] == modes[idx - 1]:
            count[modes[idx]] += 1
    return count


def plot_counts(count_1, count_2, names, filename):
    plt.figure(figsize=(16, 4))
    plt.bar(np.arange(9), count_1, alpha=0.5, label='Y1')
    plt.bar(np.arange(9), count_2, alpha=0.5, label='Y2')
    # Add value labels above each bar
    for i in range(len(names)):
        plt.text(i, count_1[i], str(count_1[i]), ha='center', va='bottom')
        plt.text(i, count_2[i], str(count_2[i]), ha='center', va='top')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(9), names)
    plt.title('Number of times each mode is repeated in each image')

    # Save the plot
    save_plot(plt, filename)

    plt.show()


def write_counts_to_file(count_modes, filename):
    with open(filename, 'w') as f:
        for count in count_modes:
            f.write(str(count) + '\n')


class VideoProcessor:
    def __init__(self, video_file, width, height):
        self.video_file = video_file
        self.width = width
        self.height = height
        self.frame_len = self.width * self.height * 3 // 2

        if not os.path.isfile(video_file):
            raise FileNotFoundError(f"{video_file} doesn't exist")

        self.f = open(video_file, 'rb')
        self.shape = (int(self.height * 1.5), self.width)
        self.endswith = os.path.splitext(video_file)[-1].lower()
        self.filename = os.path.basename(video_file).rsplit(".", 1)[0]
        self.dir = os.path.dirname(video_file)

        if self.endswith == ".yuv":
            self.ret, self.frame = self.read()
        else:
            self.ret, self.frame = self.snapshot_first_frame()

    def snapshot_first_frame(self):
        cap = cv2.VideoCapture(self.video_file)
        ret, frame = cap.read()
        if not ret:
            logging.warning("Couldn't read first frame")
        cap.release()
        return ret, frame

    def read_raw(self):
        raw = self.f.read(self.frame_len)
        if len(raw) == 0:
            return False, None
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(self.shape)
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return ret, bgr

    def close_file(self):
        self.f.close()


def scan_dir(directory):
    """
  Scans a directory and returns all files with a prefix of `.yuv`.

  Args:
    directory: The directory to scan.

  Returns:
    A list of files with a prefix of `.yuv`.
  """

    files = []
    for file in os.listdir(directory):
        if file.endswith(".yuv"):
            files.append(directory + '/' + file)

    return files


input_files = scan_dir('files/input_files')
width = 352
height = 288

videos_list = []
output_dir = 'files/output_files'
os.makedirs(output_dir, exist_ok=True)

"""
Save first frame
"""

for input_file in input_files:
    vp = VideoProcessor(input_file, width, height)
    videos_list.append(vp)

    ret, frame = vp.read()
    if ret:
        y_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
        output_file = os.path.join(output_dir, f"{vp.filename}_frame.png")
        cv2.imwrite(output_file, y_frame)
    else:
        logging.warning(f"Couldn't read frame from {input_file}")

dct_processor = DCTProcessor(qp=6)  # Create an instance of DCTProcessor with a specific qp.

for vp in videos_list:
    # Extract the first frame and convert to YUV
    ret, frame = vp.read()
    y_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]

    for qp in [6, 12, 18, 30]:
        dct_processor.qp = qp  # Update the qp for the DCTProcessor instance.

        # Create an instance of the Encoder class
        encoder = Encoder(y_frame, qp)

        # Call the full_process method of the Encoder instance
        intra, diff, diff_recon, reconstructed, modes = encoder.full_process(y_frame)
        psnr_intra = dct_processor.calculate_psnr(y_frame, intra)

        plot = plt.figure(figsize=(16, 4))

        plt.subplot(131)
        plt.imshow(diff, cmap='viridis')
        plt.title(f'Residual - MAD: {dct_processor.calculate_mad(y_frame, reconstructed)}')
        plt.xlabel('Width')
        plt.ylabel('Height')

        plt.subplot(132)
        plt.imshow(intra, cmap='viridis')
        plt.title(f'Intra Prediction - PSNR: {psnr_intra}')
        plt.xlabel('Width')
        plt.ylabel('Height')

        plt.subplot(133)
        plt.imshow(reconstructed, cmap='viridis')
        plt.title(f'Reconstructed - PSNR: {dct_processor.calculate_psnr(y_frame, reconstructed)}')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.suptitle(f'{vp.filename} - QP: {qp}')

        save_plot(plot, f'{vp.filename}_{qp}.png')

        plt.show()
        with open('results.csv', 'a') as f:
            f.write(f'{vp.filename},{qp},{psnr_intra},{dct_processor.calculate_mad(y_frame, reconstructed)}\n')

names = ["DC",
         "Verical",
         "Horizonal",
         "Diagonal down left",
         "Diagonal down right",
         "Vertical right",
         "Vertical left",
         'Horizontal down',
         'Horizontal up']

for i in range(0, len(videos_list), 2):  # iterating over pairs of videos
    vp1, vp2 = videos_list[i], videos_list[i + 1]

    count_modes_1 = count_modes_2 = np.zeros(9, dtype=int)  # Initialize the count arrays

    for vp_idx, vp in enumerate([vp1, vp2]):
        ret, frame = vp.read()
        y_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]

        for qp in [6, 12, 18, 30]:
            dct_processor.qp = qp  # Update the qp for the DCTProcessor instance.

            # Create an instance of the Encoder class
            encoder = Encoder(y_frame, qp)

            # Call the full_process method of the Encoder instance
            intra, diff, diff_recon, reconstructed, modes = encoder.full_process(y_frame)

            # Count the modes and plot the histogram
            count_modes = np.zeros(9, dtype=int)
            for idx in range(len(modes)):
                if idx == 0:
                    continue
                if modes[idx] == modes[idx - 1]:
                    count_modes[modes[idx]] += 1

            # Write counts to file
            write_counts_to_file(count_modes, f'files/output_files/modes_counts_{vp.filename}.txt')

            # Assign the modes counts to the respective variables
            if vp_idx == 0:  # if it's the first video
                count_modes_1 = count_modes
            else:  # if it's the second video
                count_modes_2 = count_modes

    # At this point we have processed both videos in the pair, so we plot the counts together
    plot_counts(count_modes_1, count_modes_2, names=names, filename=f'{vp1.filename}_{vp2.filename}_{qp}.png')
#
# psnrs_qp = {}
# modes = {2: 'DC prediction', 0: 'Vertical prediction', 1: 'Horizontal prediction', 3: 'Diagonal down-left prediction',
#          4: 'Diagonal down-right prediction', 5: 'Vertical right prediction', 6: 'Horizontal down prediction',
#          7: 'Vertical left prediction', 8: 'Horizontal up prediction'}
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # Create a file handler.
# file_handler = logging.FileHandler("logs.txt")
# file_handler.setLevel(logging.DEBUG)
#
# # Create a formatter.
# formatter = logging.Formatter("%(message)s")
# file_handler.setFormatter(formatter)
#
# # Add the file handler to the logger.
# logger.addHandler(file_handler)
#
#
# def clear_log_file():
#     """This function clears the log.txt file."""
#
#     file_path = "logs.txt"
#
#     # Open the file in write mode.
#     with open(file_path, "w") as file:
#         # Clear the file.
#         file.truncate(0)
#
#
# def pad_array_with_zeros(array):
#     """Pads all sides of an array with zeros.
#
#   Args:
#     array: The array to pad.
#
#   Returns:
#     A padded array.
#   """
#
#     padded_array = np.pad(array, (2, 2), 'constant', constant_values=0)
#     return padded_array
#
#
# def intra_prediction(block, mode, A, B, C, D, E, F, G, H, I, J, K, L, Q):
#     """
#     Performs intra prediction on a 4x4 block.
#
#     Args:
#         block: A 4x4 block of pixels.
#         mode: The intra prediction mode (0-8).
#         A, B, C, D, E, F, G, H, I, J, K, L, Q: Pixel values from surrounding blocks used for prediction.
#
#     Returns:
#         intra_predicted_block: A 4x4 block of predicted pixels.
#     """
#     if not 0 <= mode <= 8:
#         raise ValueError("Mode must be an integer between 0 and 8.")
#
#     intra_predicted_block = np.zeros((4, 4))
#
#     if mode == 2:  # DC prediction
#         P = (sum([A, B, C, D, I, J, K, L]) + 4) // 8
#         intra_predicted_block = np.full((4, 4), P)
#     elif mode == 0:  # Vertical prediction
#         intra_predicted_block = np.array([[A, B, C, D] for _ in range(4)])
#     elif mode == 1:  # Horizontal prediction
#         intra_predicted_block = np.array([[I, J, K, L] for _ in range(4)])
#     elif mode == 3:  # Diagonal down-left prediction
#         vals = [(A + 2 * B + C + 2) / 4, (B + 2 * C + D + 2) / 4, (C + 2 * D + E + 2) / 4,
#                 (D + 2 * E + F + 2) / 4, (E + 2 * F + G + 2) / 4, (F + 2 * G + H + 2) / 4,
#                 (G + 3 * H + 2) / 4]
#         arrange = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
#         for numR, row in enumerate(arrange):
#             for numC, column in enumerate(row):
#                 intra_predicted_block[numR][numC] = vals[column]
#     elif mode == 4:  # Diagonal down-right prediction
#         vals = [(L + 2 * K + J + 2) / 4, (K + 2 * J + I + 2) / 4, (J + 2 * I + Q + 2) / 4,
#                 (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4,
#                 (B + 2 * C + D + 2) / 4]
#         arrange = [[3, 4, 5, 6], [2, 3, 4, 5], [1, 2, 3, 4], [0, 1, 2, 3]]
#         for numR, row in enumerate(arrange):
#             for numC, column in enumerate(row):
#                 intra_predicted_block[numR][numC] = vals[column]
#         # return np.diag(block)
#     elif mode == 5:
#         # Vertical right prediction
#         intra_predicted_block = np.array([(Q + A + 1) / 2, (A + B + 1) / 2, (B + C + 1) / 2, (C + D + 1) / 2],
#                                          [(I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4,
#                                           (B + 2 * C + D + 2) / 4],
#                                          [(Q + 2 * I + J + 2) / 4, (Q + A + 1) / 2, (A + B + 1) / 2, (B + C + 1) / 2],
#                                          [(I + 3 * J + K + 2) / 4, (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4,
#                                           (A + 2 * B + C + 2) / 4])
#         # return np.roll(block, -1, axis=1)
#     elif mode == 6:
#         # Horizontal down prediction
#         intra_predicted_block = np.array(
#             [(Q + I + 1) / 2, (I + 2 * Q + A + 2) / 4, (Q + 2 * A + B + 2) / 4, (A + 2 * B + C + 2) / 4],
#             [(I + J + 1) / 2, (Q + 2 * I + J + 2) / 4, (Q + I + 1) / 2, (I + 2 * Q + A + 2) / 4],
#             [(J + K + 1) / 2, (I + 2 * J + K + 2) / 4, (I + J + 1) / 2, (Q + 2 * I + J + 2) / 4],
#             [(K + L + 1) / 2, (J + 2 * K + L + 2) / 4, (J + K + 1) / 2, (I + 2 * J + K + 2) / 4])
#         # return np.roll(block, -1, axis=0)
#     elif mode == 7:
#         # Vertical left prediction
#         intra_predicted_block = np.array([(A + B + 1) / 2, (B + C + 1) / 2, (C + D + 1) / 2, (D + E + 1) / 2],
#                                          [(A + 2 * B + C + 2) / 4, (B + 2 * C + D + 2) / 4, (C + 2 * D + E + 2) / 4,
#                                           (D + 2 * E + F + 2) / 4],
#                                          [(B + C + 1) / 2, (C + D + 1) / 2, (D + E + 1) / 2, (E + F + 1) / 2],
#                                          [(C + D + 1) / 2, (D + E + 1) / 2, (E + F + 1) / 2, (E + 2 * F + G + 2) / 4])
#         # return np.roll(block, 1, axis=1)
#     elif mode == 8:
#         # Horizontal up prediction
#         intra_predicted_block = np.array(
#             [(I + J + 1) / 2, (I + 2 * J + K + 2) / 4, (J + K + 1) / 2, (J + 2 * K + L + 2) / 4],
#             [(J + K + 1) / 2, (J + 2 * K + L + 2) / 4, (K + L + 1) / 2, (K + 2 * L + L + 2) / 4],
#             [(K + L + 1) / 2, (K + 2 * L + L + 2) / 4, L, L],
#             [L, L, L, L])
#         # return np.roll(block, 1, axis=0)
#     return intra_predicted_block
#
#
# def compress_image(image, qp):
#     """
#   Performs intra prediction on an image using the H.264 standard.
#
#   Args:
#       image: The image to compress.
#       qp: The quantization parameter.
#
#   Returns:
#       A compressed image.
#   """
#
#     # Convert the image to a NumPy array.
#     image_array = np.array(image)
#
#     # Perform intra prediction on the image.
#     intra_predicted_image = intra_prediction(image_array, 0)
#
#     # Perform DCT transform on the intra predicted image.
#     dct_image = cv2.dct(intra_predicted_image)
#
#     # Quantize the DCT coefficients.
#     quantized_image = dct_image / (2 ** qp)
#
#     # Perform inverse DCT transform on the quantized image.
#     decompressed_image = cv2.idct(quantized_image)
#
#     # Resize the decompressed image to 288x352.
#     resized_image = cv2.resize(decompressed_image, (288, 352))
#
#     # Return the compressed image.
#     return resized_image
#
#
# def calculate_psnr(original_image, reconstructed_image):
#     """
#   Calculates the PSNR between two images.
#
#   Args:
#     original_image: The original image.
#     reconstructed_image: The reconstructed image.
#
#   Returns:
#     The PSNR between the two images.
#   """
#
#     # Calculate the mean squared error between the two images.
#     mse = np.mean((original_image - reconstructed_image) ** 2)
#
#     # Calculate the peak signal to noise ratio.
#     psnr = 10 * np.log10(255 ** 2 / mse)
#
#     # Return the PSNR.
#     return psnr
#
#
# def display_images(original_image, intra_edited_image, residual_image, restored_image):
#     """
#   Displays the original image, the image after intra edits, the residual image, and the restored image.
#
#   Args:
#     original_image: The original image.
#     intra_edited_image: The image after intra edits.
#     residual_image: The residual image.
#     restored_image: The restored image.
#   """
#
#     # Display the original image.
#     cv2.imshow("Original Image", original_image)
#
#     # Display the image after intra edits.
#     cv2.imshow("Intra Edited Image", intra_edited_image)
#
#     # Display the residual image.
#     cv2.imshow("Residual Image", residual_image)
#
#     # Display the restored image.
#     cv2.imshow("Restored Image", restored_image)
#
#     # Wait for the user to press a key.
#     cv2.waitKey(0)
#
#     # Close all the windows.
#     cv2.destroyAllWindows()
#
#
# def scan_dir(directory):
#     """
#   Scans a directory and returns all files with a prefix of `.yuv`.
#
#   Args:
#     directory: The directory to scan.
#
#   Returns:
#     A list of files with a prefix of `.yuv`.
#   """
#
#     files = []
#     for file in os.listdir(directory):
#         if file.endswith(".yuv"):
#             files.append(directory + '/' + file)
#
#     return files
#
#
# def calculate_mad(residual_image, luminance_matrix):
#     """
#   Calculates the MAD (Mean Absolute Difference) between the original image and the residual image.
#
#   Args:
#     residual_image: The residual image.
#     luminance_matrix: The luminance image
#
#   Returns:
#     The MAD between the original image and the residual image.
#   """
#
#     # Calculate the mean absolute difference between the original image and the residual image.
#     diff = np.abs(residual_image - luminance_matrix)
#     mad = np.mean(diff)
#
#     # Return the MAD.
#     return mad
#
#
# def calculate_residual_image(intra_predicted_image, luminance_matrix):
#     """
#   Calculates the residual image between the intra predicted image and the original image.
#
#   Args:
#     intra_predicted_image: The intra predicted image.
#     luminance_matrix: The original image.
#
#   Returns:
#     The residual image.
#   """
#
#     # Calculate the difference between the intra predicted image and the original image.
#     residual_image = luminance_matrix - intra_predicted_image
#
#     # Return the residual image.
#     return residual_image
#
#
# def sad(source, predicted):
#     """Computes the sum of absolute differences between the source and predicted images."""
#     sad = 0
#     for i in range(len(source)):
#         for j in range(len(source[0])):
#             sad += abs(source[i][j] - predicted[i][j])
#     return sad
#
#
# def pad_array(array, value, pad_width):
#     """Pads an array with a certain value."""
#     padded_array = np.pad(array, pad_width, mode='constant', constant_values=value)
#     return padded_array
#
#
# def divide_array_into_blocks(array):
#     """Divides a NumPy array into 4x4 blocks.
#
#   Args:
#     array: The NumPy array to divide.
#
#   Returns:
#     A list of 4x4 NumPy arrays.
#   """
#
#     blocks = []
#     height = array.shape[0]
#     width = array.shape[1]
#     for i in range(0, height, 4):
#         for j in range(0, width, 4):
#             block = array[i:i + 4, j:j + 4]
#             blocks.append(block)
#     return blocks
#
#
# def find_block_index(array, block_size, block_value):
#     """Finds a block inside an array and returns the row and column it starts at."""
#     block_found = False
#     row = None
#     column = None
#     for i in range(len(array) - block_size + 1):
#         for j in range(len(array[0]) - block_size + 1):
#             if np.array_equal(array[i:i + block_size, j:j + block_size], block_value):
#                 block_found = True
#                 row = i
#                 column = j
#                 break
#         if block_found:
#             break
#     return row, column
#
#
# def ABCDEFGHIJKLQ(padded, row, column):
#     A = padded[row - 1][column]
#     B = padded[row - 1][column + 1]
#     C = padded[row - 1][column + 2]
#     D = padded[row - 1][column + 3]
#     E = padded[row - 1][column + 4]
#     F = padded[row - 1][column + 5]
#     G = padded[row - 1][column + 6]
#     H = padded[row - 1][column + 7]
#     I = padded[row][column - 1]
#     J = padded[row + 1][column - 1]
#     K = padded[row + 2][column - 1]
#     L = padded[row + 3][column - 1]
#     Q = padded[row - 1][column - 1]
#     return A, B, C, D, E, F, G, H, I, J, K, L, Q
#
#
# def numpy_array_to_image(array):
#     """Converts a NumPy array to an image.
#
#   Args:
#     array: The NumPy array to convert.
#
#   Returns:
#     The image.
#   """
#
#     image = PIL.Image.fromarray(array, 'RGB')
#     return image
#
#
# def main():
#     """
#     1. Read two videos and extract one uncompressed Intensity (Luma) image from each video.
#     """
#     videos_list = []
#     input_files = scan_dir('files/input_files')
#     width = 352
#     height = 288
#     global psnrs_qp
#     """
#     2. a. Perform intra prediction on the first Intensity(Luma) image from each video.
#     """
#
#     for input_file in input_files:
#
#         videos_list.append(VideoProcessor(input_file, width, height))
#         size = (288, 352)  # 352x288
#
#         ret, frame = videos_list[-1].read()
#         imageframe = frame
#
#         # Extracting the Y channel
#         y1 = cv2.cvtColor(imageframe, cv2.COLOR_BGR2YUV)[:, :, 0]
#
#         # Create a subfolder named 'output' if it doesn't exist
#         if not os.path.exists('files/output_files'):
#             os.makedirs('files/output_files')
#
#         # Save the images to disk
#         cv2.imwrite('files/output_files/' + videos_list[-1].filename + '_frame.png', y1)
#     """
#     2. b. For each selected image and QP values of 6, 12, 18, and 30, display the original image, the image after intra edits, the residual image, and the restored image.
#     Calculate the PSNR between the original images and the reconstructed images. Write the appropriate PSNR value in the title of the restored image.
#
#     c. Calculate the MAD (Mean Absolute Difference) that can be extracted from the process of searching for the best modes.
#     """
#     # Perform intra prediction on the first Intensity(Luma) image from each video
#     for input_file in input_files:
#         file_name = input_file.split('/')[-1]
#         luminance_matrix = np.load(f"files/output_files/{file_name}_luminance.npy")
#         bus_intra_prediction = intra_prediction(luminance_matrix, 0)
#         # print(bus_intra_prediction)
#
#         # Save the intra predicted images to a file
#         if not os.path.exists("files/intra_predictions"):
#             os.makedirs("files/intra_predictions")
#         np.save(f"files/intra_predictions/{file_name}_intra_prediction.npy", bus_intra_prediction)
#
#         # Save the intra predicted images to a png file
#         cv2.imwrite(f"intra_predictions/{file_name}_intra_prediction.png", bus_intra_prediction)
#         psnrs_qp[file_name] = {}
#         for qp in [6, 12, 18, 30]:
#             compress_img = compress_image(bus_intra_prediction, qp)
#             # Calculate the PSNR between the original and intra predicted images.
#             psnr = calculate_psnr(luminance_matrix, compress_img.T)
#             print(f'{file_name} psnr is {psnr:.2f} with QP={qp}')
#             psnrs_qp[file_name][qp] = psnr
#             # Display the original, intra predicted, and restored images.
#             residual_image = calculate_residual_image(bus_intra_prediction, luminance_matrix)
#             mad = calculate_mad(residual_image, luminance_matrix)
#             print(mad, 'qp', qp)
#             # Save the original, intra predicted, and restored images to the files/output_files directory
#             cv2.imwrite(f"files/output_files/{file_name}_{qp}_original.png", luminance_matrix)
#             cv2.imwrite(f"files/output_files/{file_name}_{qp}_intra_prediction.png", bus_intra_prediction)
#             cv2.imwrite(f"files/output_files/{file_name}_{qp}_restored.png", compress_img)
#
#
# def combine_blocks_into_array(blocks):
#     """Combines a list of 4x4 blocks into a single NumPy array.
#
#   Args:
#     blocks: A list of 4x4 NumPy arrays.
#
#   Returns:
#     A single NumPy array.
#   """
#
#     array = np.zeros((blocks.shape[0] * 4, blocks.shape[1] * 4))
#     for i in range(blocks.shape[0]):
#         for j in range(blocks.shape[1]):
#             array[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = blocks[i, j]
#
#     return array
#
#
# if __name__ == "__main__":
#     # s = "$92 & 91 & 89 & 86 & 85 & 85 & 87 & 90 & 94 & 94 & 94 & 94 & 93 & 96 & 97 & 95 \\ 91 & 90 & 89 & 87 & 86 & 85 & 88 & 92 & 94 & 95 & 95 & 96 & 95 & 96 & 99 & 96 \\ 89 & 89 & 89 & 90 & 87 & 84 & 85 & 89 & 93 & 94 & 96 & 97 & 98 & 99 & 99 & 97 \\ 88 & 88 & 89 & 92 & 91 & 85 & 85 & 88 & 91 & 93 & 95 & 97 & 99 & 102 & 99 & 98 \\ 88 & 90 & 93 & 94 & 100 & 97 & 89 & 83 & 81 & 84 & 93 & 98 & 104 & 107 & 102 & 96 \\ 96 & 98 & 98 & 95 & 96 & 100 & 97 & 89 & 79 & 82 & 89 & 101 & 105 & 106 & 104 & 97 \\ 95 & 91 & 96 & 97 & 94 & 92 & 95 & 94 & 89 & 83 & 89 & 93 & 91 & 95 & 99 & 99 \\ 90 & 85 & 83 & 87 & 85 & 88 & 87 & 89 & 93 & 91 & 82 & 83 & 84 & 89 & 91 & 93 \\ 87 & 89 & 90 & 89 & 88 & 86 & 84 & 80 & 85 & 87 & 87 & 85 & 84 & 84 & 84 & 86 \\ 86 & 85 & 87 & 90 & 90 & 89 & 91 & 84 & 85 & 86 & 87 & 88 & 86 & 82 & 81 & 83 \\ 86 & 85 & 85 & 88 & 87 & 90 & 89 & 87 & 87 & 86 & 87 & 90 & 90 & 89 & 91 & 93 \\ 86 & 88 & 87 & 84 & 83 & 83 & 87 & 85 & 88 & 87 & 87 & 88 & 89 & 90 & 90 & 88 \\ 88 & 84 & 85 & 81 & 83 & 84 & 87 & 90 & 91 & 91 & 92 & 93 & 90 & 84 & 81 & 79 \\ 85 & 83 & 83 & 81 & 79 & 80 & 83 & 86 & 87 & 91 & 95 & 95 & 96 & 86 & 82 & 82 \\ 85 & 84 & 84 & 84 & 85 & 85 & 87 & 89 & 90 & 93 & 96 & 98 & 97 & 97 & 87 & 86 \\ 87 & 87 & 87 & 88 & 87 & 87 & 89 & 91 & 91 & 96 & 100 & 100 & 100 & 98 & 94 & 89$"
#     # for char in s:
#     #     if char == "$":
#     #         print("[", end='')
#     #     elif char == "&":
#     #         print(",", end='')
#     #     elif char == ("\\"):
#     #         print("],\n[", end='')
#     #     else:
#     #         print(char, end='')
#     main()
#     example = np.array([[92, 91, 89, 86, 85, 85, 87, 90, 94, 94, 94, 94, 93, 96, 97, 95],
#                         [91, 90, 89, 87, 86, 85, 88, 92, 94, 95, 95, 96, 95, 96, 99, 96],
#                         [89, 89, 89, 90, 87, 84, 85, 89, 93, 94, 96, 97, 98, 99, 99, 97],
#                         [88, 88, 89, 92, 91, 85, 85, 88, 91, 93, 95, 97, 99, 102, 99, 98],
#                         [88, 90, 93, 94, 100, 97, 89, 83, 81, 84, 93, 98, 104, 107, 102, 96],
#                         [96, 98, 98, 95, 96, 100, 97, 89, 79, 82, 89, 101, 105, 106, 104, 97],
#                         [95, 91, 96, 97, 94, 92, 95, 94, 89, 83, 89, 93, 91, 95, 99, 99],
#                         [90, 85, 83, 87, 85, 88, 87, 89, 93, 91, 82, 83, 84, 89, 91, 93],
#                         [87, 89, 90, 89, 88, 86, 84, 80, 85, 87, 87, 85, 84, 84, 84, 86],
#                         [86, 85, 87, 90, 90, 89, 91, 84, 85, 86, 87, 88, 86, 82, 81, 83],
#                         [86, 85, 85, 88, 87, 90, 89, 87, 87, 86, 87, 90, 90, 89, 91, 93],
#                         [86, 88, 87, 84, 83, 83, 87, 85, 88, 87, 87, 88, 89, 90, 90, 88],
#                         [88, 84, 85, 81, 83, 84, 87, 90, 91, 91, 92, 93, 90, 84, 81, 79],
#                         [85, 83, 83, 81, 79, 80, 83, 86, 87, 91, 95, 95, 96, 86, 82, 82],
#                         [85, 84, 84, 84, 85, 85, 87, 89, 90, 93, 96, 98, 97, 97, 87, 86],
#                         [87, 87, 87, 88, 87, 87, 89, 91, 91, 96, 100, 100, 100, 98, 94, 89]])
#
#     image = Image.open('files/output_files/bus_cif_6_original.png')
#     example = np.asarray(image)
#
#     example_padded = pad_array(example, 128, 4)
#     blocks = divide_array_into_blocks(example)
#
#     first_top_left = np.array([[92, 91, 89, 86],
#                                [91, 90, 88, 86],
#                                [89, 89, 89, 88],
#                                [89, 87, 88, 93]])
#     predicted_images = []
#     residual_blocks = []
#     for blockNum, block in enumerate(blocks):
#         try:
#             row, column = find_block_index(example_padded, 4, block)
#             A, B, C, D, E, F, G, H, I, J, K, L, Q = ABCDEFGHIJKLQ(example_padded, row, column)
#             SAD_value = [np.inf for _ in range(9)]
#             for mode in range(9):
#                 try:
#                     predicted_block = intra_prediction(first_top_left, mode, A, B, C, D, E, F, G, H, I, J, K, L, Q)
#                     SAD_value[mode] = sad(block, predicted_block)
#                 except:
#                     continue
#             print(f'block #{blockNum}: best mode is {modes[SAD_value.index(min(SAD_value))]} with SAD {min(SAD_value)}')
#         except:
#             print(f'block #{blockNum}: Errored')
#             continue
#         predicted_block = intra_prediction(first_top_left, SAD_value.index(min(SAD_value)), A, B, C, D, E, F, G, H, I,
#                                            J, K, L, Q)
#         predicted_images.append(predicted_block)
#         residual_block = calculate_residual_image(predicted_block, block)
#         residual_blocks.append(residual_block)
#         mad = calculate_mad(residual_block, block)
#     contracted_image = combine_blocks_into_array(predicted_images)
#     residual = combine_blocks_into_array(residual_blocks)
#
#     qp = 6
#     # Perform DCT transform on the intra predicted image.
#     dct_image = cv2.dct(contracted_image)
#
#     # Quantize the DCT coefficients.
#     quantized_image = dct_image / (2 ** qp)
#
#     # Perform inverse DCT transform on the quantized image.
#     decompressed_image = cv2.idct(quantized_image)
#
#     # Resize the decompressed image to 288x352.
#     resized_image = cv2.resize(decompressed_image, (288, 352))
#     image = numpy_array_to_image(resized_image)
#
#     # Save the image to the directory '/tmp/images'.
#     image.save('resizedimage.jpg')
#     image = numpy_array_to_image(example)
#
#     # Save the image to the directory '/tmp/images'.
#     image.save('exampleimage.jpg')
#     clear_log_file()
#     exit()
#     main()
#
#     # Log the dictionary.
#     logger.debug(f"psnrs_qp = {psnrs_qp}")
