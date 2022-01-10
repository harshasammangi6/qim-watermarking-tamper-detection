import numpy as np

def detect_watermark(image, shape, step=10, mode='binary'):
    h, w = shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = image[i, j]
            bit = (pixel % step) > (step // 2)
            extracted[i, j] = 255 if bit else 0
    return extracted
