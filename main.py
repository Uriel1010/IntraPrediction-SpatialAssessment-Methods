import json
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

results = {}


for vp in videos_list:
    # Extract the first frame and convert to YUV
    ret, frame = vp.read()
    y_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]

    # Initialize the key in the dictionary if it does not exist
    if vp.filename not in results:
        results[vp.filename] = {}

    for qp in [6, 12, 18, 30]:
        dct_processor.qp = qp  # Update the qp for the DCTProcessor instance.

        # Create an instance of the Encoder class
        encoder = Encoder(y_frame, qp)

        # Call the full_process method of the Encoder instance
        intra, diff, diff_recon, reconstructed, modes = encoder.full_process(y_frame)
        psnr_intra = dct_processor.calculate_psnr(y_frame, intra)

        fig1, ax1 = plt.subplots(figsize=(2.5, 2))
        ax1.imshow(diff)
        ax1.set_title(f'Residual')# - MAD: {dct_processor.calculate_mad(y_frame, reconstructed)}')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        fig1.savefig(f'files/output_files/{vp.filename}_{qp}_residual.png', dpi=300)

        fig2, ax2 = plt.subplots(figsize=(2.5, 2))
        ax2.imshow(intra)
        ax2.set_title(f'Intra Prediction')# - PSNR: {psnr_intra}')
        ax2.set_xlabel('Width')
        ax2.set_ylabel('Height')
        fig2.savefig(f'files/output_files/{vp.filename}_{qp}_intra.png', dpi=300)

        fig3, ax3 = plt.subplots(figsize=(2.5, 2))
        ax3.imshow(reconstructed)
        ax3.set_title(f'Reconstructed')# - PSNR: {dct_processor.calculate_psnr(y_frame, reconstructed)}')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
        fig3.savefig(f'files/output_files/{vp.filename}_{qp}_reconstructed.png', dpi=300)


        psnr_intra = dct_processor.calculate_psnr(y_frame, intra)
        mad = dct_processor.calculate_mad(y_frame, reconstructed)

        results[vp.filename][qp] = {
            "Residual_MAD": mad,
            "Intra_Prediction_PSNR": psnr_intra,
            "Reconstructed_PSNR": dct_processor.calculate_psnr(y_frame, reconstructed)
        }

        with open('results.csv', 'a') as f:
            f.write(f'{vp.filename},{qp},{psnr_intra},{dct_processor.calculate_mad(y_frame, reconstructed)}\n')

# Save the results dictionary to a JSON file
with open('files/output_files/results.json', 'w') as f:
    json.dump(results, f, indent=4)

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
