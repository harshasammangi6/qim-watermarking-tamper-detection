from embed_qim import embed_watermark
from detect_qim import detect_watermark
import cv2
import os

def run():
    image = cv2.imread('data/test_image.jpg', 0)
    watermark = cv2.imread('data/binary_watermark.png', 0)
    if image is None or watermark is None:
        print("Missing input files.")
        return
    watermarked = embed_watermark(image, watermark, mode='binary')
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/watermarked.jpg', watermarked)
    extracted = detect_watermark(watermarked, watermark.shape, mode='binary')
    cv2.imwrite('results/extracted.png', extracted)

if __name__ == '__main__':
    run()
