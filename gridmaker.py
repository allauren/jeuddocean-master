import cv2
import numpy as np
import json
import operator
import requests


# fonction qui permet de retourner la proximite du code BGR par rapport aux couleurs de references
def color_difference (color1, color2):
    return sum([abs(component1-component2) for component1, component2 in zip(color1, color2)])

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

def create_json (color1, color2, color3):
    colors = {"Red": (243, 54, 191), "Yellow": (0, 255, 255), "Green": (80, 255, 149), "White": (255, 255, 255), "Blue": (255, 0, 0)}
    poisson = 0
    eolien = 0
    loisir = 0
    transport = 0

    values =[]
    diff = [[color_difference(color1, target_value), target_name] for target_name, target_value in colors.items()]
    diff.sort()
    values.append(diff[0][1])
    diff2 = [[color_difference(color2, target_value), target_name] for target_name, target_value in colors.items()]
    diff2.sort()
    values.append(diff2[0][1])
    diff3 = [[color_difference(color3, target_value), target_name] for target_name, target_value in colors.items()]
    diff3.sort()
    values.append(diff3[0][1])
    for value in values:
        if value == "Red":
            poisson += 1
        elif value == "White":
            eolien += 1
        elif value == "Vert":
            loisir += 1
        elif value == "Yellow":
            transport += 1
    return ({
        "peche": poisson,
        "eolien": eolien,
        "loisir": loisir,
        ²"transport": transport
    })




def gridparse():
    #cv2.VideoCapture()
    clean_img = cv2.imread('map.png')
    while(1):
        img = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
        processed = pre_process_image(img)
        corners = find_corners_of_largest_polygon(processed)
        cropped = crop_and_warp(clean_img, corners)
        # cv2.imwrite('hope.jpg', cropped)
        # calculs pour determiner la position des cercles par rapport a la taille du carre redessine

        size = cropped.shape[0]
        height_start = [int(size * 370 / 4219), int(size * 530 / 4219)]
        width_start = [int(size * 450 / 4219), int(size * 360 / 4219), int(size * 550 / 4219)]
        thick = int(size * 155 / 4219)
        change = int(size * 632 / 4219)
        x = [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
       request = {}
        cell = 1
        col = 0
        for y in x :
            valy = col * change
            print(col)
            for i in y :
                valx =  i * change
                color1 = unique_count_app(cropped[height_start[0] + valx : height_start[0] + valx + thick, width_start[0] + valy :width_start[0] + valy + thick])
                color2 = unique_count_app(cropped[height_start[1] + valx : height_start[1] + valx + thick, width_start[1] + valy :width_start[1] + valy + thick])
                color3 = unique_count_app(cropped[height_start[1] + valx : height_start[1] + valx + thick, width_start[2] + valy :width_start[2] + valy + thick])
                dic = {cell:create_json(color1, color2, color3)}
                request.update(dic)
                cell += 1
                print(cell)
            col += 1
        url = "http://localhost:" #completer ici
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        requests.post(url, json=dumps(request, sort_keys=True, indent=4), headers=headers)



if __name__ == '__main__':
    gridparse()

