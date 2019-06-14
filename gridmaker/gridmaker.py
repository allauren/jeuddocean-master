import cv2
import numpy as np
from matplotlib import pyplot as plt


def horizontal_image(proc):
    horizontal = np.copy(proc)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite("horizontal.png", horizontal)
    return horizontal


def vertical_image(proc):
    vertical = np.copy(proc)
    rows = vertical.shape[0]
    verticalsize = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    cv2.imwrite("vertical.png", vertical)
    return vertical


def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vertical = vertical_image(proc)
        horizontal = horizontal_image(proc)
        proc = cv2.addWeighted(vertical, 1, horizontal, 1, 0.0)
        cv2.imwrite('proc.png', proc)
    return proc


def main():
    img = cv2.imread('Test1.png', cv2.IMREAD_GRAYSCALE)
    processed = pre_process_image(img)
    cv2.imwrite('thresholdprocess.png', processed)
    for values in processed:
        for value in values:
            print(value, end=""),
    print()
    contours, heir = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    all_contours = cv2.drawContours(processed.copy(), contours, -1, (255, 0, 0), 2)
    cv2.imwrite('lololo.jpg', all_contours)


if __name__ == '__main__':
    main()
