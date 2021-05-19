import math
from points import redPoints
import cv2 as cv

def filledRect(h,img, point, width, height, color):
    """
    draws a filled rect with given 4 points
    """

    for w in range(0,int(width),1):
        s = math.sqrt(w**2 + height**2)
        delta = math.acos(height/s)*180/math.pi
        points = redPoints((point[0][0], point[0][1]), height, w, s, point[1], delta)
        cv.line(img, (int(points[0][0]), int(points[0][1])), (int(points[2][0]), int(points[2][1])), color, 2)
        cv.line(img, (int(points[1][0]), int(points[1][1])), (int(points[3][0]), int(points[3][1])), color, 2)
        cv.line(h, (int(points[0][0]), int(points[0][1])), (int(points[2][0]), int(points[2][1])), 0, 2)
        cv.line(h, (int(points[1][0]), int(points[1][1])), (int(points[3][0]), int(points[3][1])), 0, 2)