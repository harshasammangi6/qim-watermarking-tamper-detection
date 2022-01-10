import numpy as np

def embed_watermark(image, watermark, step=10, mode='binary'):
    h, w = watermark.shape
    output = image.copy()
    for i in range(h):
        for j in range(w):
            wm_bit = watermark[i, j] > 127 if mode == 'binary' else watermark[i, j] / 255.0
            pixel = image[i, j]
            q = int(pixel / step)
            output[i, j] = step * q + (step // 2 if wm_bit else 0)
    return output
