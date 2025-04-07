import numpy as np

def detect_watermark(image, shape, step=10, mode='binary'):
    """
    Detects a watermark from an image using Quantization Index Modulation (QIM).

    Parameters:
        image (ndarray): Input image as a 2D numpy array.
        shape (tuple): Shape of the watermark (height, width).
        step (int): Quantization step size. Default is 10.
        mode (str): Mode of detection (currently unused). Default is 'binary'.

    Returns:
        ndarray: Extracted watermark as a binary image.
    """
    h, w = shape
    extracted = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            pixel = image[i, j]
            extracted[i, j] = 255 if (pixel % step) > (step // 2) else 0

    return extracted