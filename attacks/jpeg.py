import cv2

def jpeg_compress(image, path, quality=40):
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
