import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def intra_prediction(block, mode):
  """
  Performs intra prediction on a 4x4 block.

  Args:
    block: A 4x4 block of pixels.
    mode: The intra prediction mode.

  Returns:
    A 4x4 block of predicted pixels.
  """

  if mode == 0:
    # DC prediction
    return np.mean(block, axis=0)
  elif mode == 1:
    # Vertical prediction
    return block[:, -1].reshape(-1, 1)
  elif mode == 2:
    # Horizontal prediction
    return block[-1, :]
  elif mode == 3:
    # Diagonal down-left prediction
    return np.diag(block[::-1])
  elif mode == 4:
    # Diagonal down-right prediction
    return np.diag(block)
  elif mode == 5:
    # Vertical right prediction
    return np.roll(block, -1, axis=1)
  elif mode == 6:
    # Horizontal down prediction
    return np.roll(block, -1, axis=0)
  elif mode == 7:
    # Vertical left prediction
    return np.roll(block, 1, axis=1)
  elif mode == 8:
    # Horizontal up prediction
    return np.roll(block, 1, axis=0)

def main():
  input_files = ["bus_cif", "coastguard_cif"]
  width = 352
  height = 288

  for input_file in input_files:
    with open(input_file + ".yuv", "rb") as file:
      # Read the first luminance image from the file.
      luminance_image = np.fromfile(file, dtype=np.uint8, count=width * height)

      # Reshape the luminance image to a 2D matrix
      luminance_matrix = np.reshape(luminance_image, (height, width))

      # Save the luminance image to an uncompressed file.
      np.save(f"{input_file}_luminance.npy", luminance_matrix)
      cv2.imwrite(f"{input_file}_luminance.png", luminance_matrix)

  # Perform intra prediction on the first Intensity(Luma) image from each video
  for input_file in input_files:
    luminance_matrix = np.load(f"{input_file}_luminance.npy")
    bus_intra_prediction = intra_prediction(luminance_matrix, 0)

    # Save the intra predicted images to a file
    if not os.path.exists("files/intra_predictions"):
      os.makedirs("files/intra_predictions")
    np.save(f"files/intra_predictions/{input_file}_intra_prediction.npy", bus_intra_prediction)

    # Save the intra predicted images to a png file
    cv2.imwrite(f"intra_predictions/{input_file}_intra_prediction.png", bus_intra_prediction)


if __name__ == "__main__":
  main()
