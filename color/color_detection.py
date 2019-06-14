import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

#
# Parsing
#

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video")
args = vars(ap.parse_args())

#
# Lecture de video
#

video = cv2.VideoCapture('./videoplayback.mp4')
if (video.isOpened() == False):
    print('Error while opening video ...')

while (video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        #      for (lower, upper) in limits:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 50, 20), (5, 255, 255))
        mask2 = cv2.inRange(hsv, (175, 50, 20), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        croped = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('video is', frame)
        cv2.imshow('mask is', mask)
        cv2.imshow('croped is', croped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exiting ...')
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
