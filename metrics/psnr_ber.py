import numpy as np
import math

def compute_psnr_ber(original, extracted):
    mse = np.mean((original - extracted) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    total = original.size
    errors = np.count_nonzero(original != extracted)
    ber = errors / total

    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bit Error Rate: {ber:.6f}")
