import cv2
import numpy as np
import argparse
import imutils
import json
from matplotlib import pyplot as plt
import sys
from cmath import isclose
import os, shutil
import operator
from sklearn.cluster import KMeans



def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

    # fonction qui permet de retrouver la couleur la plus presente dans une zone donnee retour en BGR peut etre faudra
    # t'il convertir cette valeur en "Couleur" texte afin de faciliter son identification par la suite
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

    # cette fonction permet de redefinir une zone d'interet a partir des points qui ont ete determines comme appartenant
    # au plus grand carre dans l'image et de "redresser" l'image afin d'obtenir une carte qui soit exploitable facilement.

def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
        ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

    #cette fonction permet de determiner la position des coins du polygone le plus grand, cette fonctionnalite permet
    #par la suite de redessiner sur le champ de la camera le polygone qui nous interesse et d'appliquer une regle de proportionnalite
    #qui determine la position de chaque cercle par rapport a la longieur et largeur de ce polygone trouve.

def find_corners_of_largest_polygon(img):
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    # cette fonction permet de transformer l'image que nous possedons actuellement en une image noir et blanc afin
    # par la suite de determiner les contours de la carte ce qui nous permetra de determiner les couleurs presentes sur les cases.

def pre_process_image(img):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    return proc

def main():
    clean_img = cv2.imread('map.png')
    img = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
    processed = pre_process_image(img)
    corners = find_corners_of_largest_polygon(processed)
    cropped = crop_and_warp(clean_img, corners)
    # cv2.imwrite('hope.jpg', cropped)
    # calculs pour determiner la position des cercles par rapport a la taille du carre redessine

    size = cropped.shape[0]
    height_start = int(size * 400 / 4219)
    print(height_start)

    cv2.imwrite('ball.jpg', cropped[height_start:530, 450:600])
    cv2.imwrite('ball2.jpg', cropped[530:700, 350:510])
    cv2.imwrite('ball3.jpg', cropped[530:700, 550:700])

    val1 = unique_count_app(cropped[350:530, 450:600])
    val2 =unique_count_app(cropped[530:700, 350:510])
    val3 = unique_count_app(cropped[530:700, 550:700])
    a = {'circle 1':str(val1),
         'circle 2':str(val2),
         'circle 3':str(val3)}


    aa = json.dumps(a)
    print(aa)
    return


if __name__ == '__main__':
    main()




def clean_files ():
    folder = 'blocks'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def horizontal_image(proc):
    horizontal = np.copy(proc)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3))
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite("horizontal.png", horizontal)
    return horizontal


def vertical_image(proc):
    vertical = np.copy(proc)
    rows = vertical.shape[0]
    verticalsize = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.dilate(vertical, verticalStructure)
    cv2.imwrite("vertical.png", vertical)

    return vertical



def find_boxes(contours, img):
    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if isclose(w, h, rel_tol=0.10):
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cv2.imwrite('blocks/' + 'file number' +str(idx) + '.png', new_img)

