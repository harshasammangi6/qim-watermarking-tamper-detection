import numpy as np

def embed_watermark(image, watermark, step=10, mode='binary'):
    """
    Embeds a watermark into an image using Quantization Index Modulation (QIM).

    Parameters:
        image (ndarray): Input image as a 2D numpy array.
        watermark (ndarray): Watermark to embed as a 2D numpy array.
        step (int): Quantization step size. Default is 10.
        mode (str): Embedding mode ('binary' or 'grayscale'). Default is 'binary'.

    Returns:
        ndarray: Image with the embedded watermark.
    """
    if mode not in ['binary', 'grayscale']:
        raise ValueError("Unsupported mode. Use 'binary' or 'grayscale'.")

    h, w = watermark.shape
    output = image.copy()

    for i in range(h):
        for j in range(w):
            wm_value = watermark[i, j]
            wm_bit = 1 if wm_value > 127 else 0 if mode == 'binary' else wm_value / 255.0

            pixel = image[i, j]
            q = pixel // step
            output[i, j] = step * q + (step // 2 if wm_bit else 0)

    return output