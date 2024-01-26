import cv2

def bilateral_filter(image, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)