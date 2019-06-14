import cv2
import numpy as np
import argparse
import imutils
from matplotlib import pyplot as plt


def sort_contours(cnts, method="right-to-left"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


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
    return proc


def find_boxes(contours, img):
    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)
        idx += 1
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite('blocks/' + str(idx) + '.png', new_img)


def main():
    img = cv2.imread('Test1.png', cv2.IMREAD_GRAYSCALE)
    processed = pre_process_image(img)
    cv2.imwrite('processed.png', processed)

    contours, tresh = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    all_contours = cv2.drawContours(processed.copy(), contours, -1, (128, 0, 128), 3)
    cv2.imwrite('lololo.jpg', all_contours)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    find_boxes(contours, img)


if __name__ == '__main__':
    main()
