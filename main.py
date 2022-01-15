from embed_qim import embed_watermark
from detect_qim import detect_watermark
from metrics.psnr_ber import compute_psnr_ber
from attacks.jpeg import jpeg_compress
import cv2, os

def run():
    image = cv2.imread('data/test_image.jpg', 0)
    watermark = cv2.imread('data/grayscale_watermark.png', 0)
    if image is None or watermark is None:
        print("Missing input files.")
        return
    print("Embedding watermark...")
    watermarked = embed_watermark(image, watermark, step=10, mode='grayscale')
    cv2.imwrite('results/watermarked.jpg', watermarked)

    print("Simulating JPEG compression...")
    jpeg_path = 'results/watermarked_compressed.jpg'
    jpeg_compress(watermarked, jpeg_path)
    comp_img = cv2.imread(jpeg_path, 0)

    print("Extracting watermark...")
    extracted = detect_watermark(comp_img, watermark.shape, step=10, mode='grayscale')
    cv2.imwrite('results/extracted.jpg', extracted)

    print("Evaluating results...")
    compute_psnr_ber(watermark, extracted)

if __name__ == '__main__':
    run()
