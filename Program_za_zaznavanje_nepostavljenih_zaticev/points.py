import math
import cv2 as cv

def redPoints(center, height, width,s, fi, delta):
    """calculates and returns a list of red points used for detection"""

    # red points
    sin_fi_de = math.sin((fi + delta)*math.pi/180)
    cos_fi_de = math.cos((fi + delta)*math.pi/180)
    sin_fi_mde = math.sin((fi - delta)*math.pi/180)
    cos_fi_mde = math.cos((fi - delta)*math.pi/180)

    X_TL = center[0] + s * sin_fi_mde
    Y_TL = center[1] + s * cos_fi_mde

    X_TR = center[0] + s * sin_fi_de
    Y_TR = center[1] + s * cos_fi_de

    X_BL = center[0] - s * sin_fi_de
    Y_BL = center[1] - s * cos_fi_de

    X_BR = center[0] - s * sin_fi_mde
    Y_BR = center[1] - s * cos_fi_mde

    return [ [X_TL, Y_TL], [X_TR, Y_TR], [X_BL, Y_BL], [X_BR, Y_BR]]


def darkBluePoints(center, height, width, fi):
    """ calculates and returns a list of dark blue points used for detection """

     # dark blue
    sin_fi = math.sin(fi*math.pi/180)
    cos_fi = math.cos(fi*math.pi/180)


    X_TOP = center[0] + height * sin_fi
    Y_TOP = center[1] + height * cos_fi

    X_BOT = center[0] - height * sin_fi
    Y_BOT = center[1] - height * cos_fi

    return [ [X_TOP, Y_TOP], [X_BOT, Y_BOT]]




def yellowPoints(center, height, width,fi):
    """calculates and returns a list of yellow points used for detection"""

      # yellow points
    s = math.sqrt(width**2 + (height * 2/3)**2)
    delta = math.acos(2/3*height/s)*180/math.pi
    sin_fi_de = math.sin((fi + delta)*math.pi/180)
    cos_fi_de = math.cos((fi + delta)*math.pi/180)
    sin_fi_mde = math.sin((fi - delta)*math.pi/180)
    cos_fi_mde = math.cos((fi - delta)*math.pi/180)

    X_TL = center[0] + s * sin_fi_mde
    Y_TL = center[1] + s * cos_fi_mde

    X_TR = center[0] + s * sin_fi_de
    Y_TR = center[1] + s * cos_fi_de

    X_BL = center[0] - s * sin_fi_de
    Y_BL = center[1] - s * cos_fi_de

    X_BR = center[0] - s * sin_fi_mde
    Y_BR = center[1] - s * cos_fi_mde

    return [[X_TL, Y_TL], [X_TR, Y_TR], [X_BL, Y_BL], [X_BR, Y_BR]]


def pinkPoints(center, height, width, fi):
    """calculates and returns a list of pink points used for detection"""

        # pink points
    s = math.sqrt((2/3*width)**2 + (1/3*height)**2)
    delta = math.acos(1/3*height/s)*180/math.pi
    sin_fi_de = math.sin((fi + delta)*math.pi/180)
    cos_fi_de = math.cos((fi + delta)*math.pi/180)
    sin_fi_mde = math.sin((fi - delta)*math.pi/180)
    cos_fi_mde = math.cos((fi - delta)*math.pi/180)

    X_TL = center[0] + s * sin_fi_mde
    Y_TL = center[1] + s * cos_fi_mde

    X_TR = center[0] + s * sin_fi_de
    Y_TR = center[1] + s * cos_fi_de

    X_BL = center[0] - s * sin_fi_de
    Y_BL = center[1] - s * cos_fi_de

    X_BR = center[0] - s * sin_fi_mde
    Y_BR = center[1] - s * cos_fi_mde

    return [[X_TL, Y_TL], [X_TR, Y_TR], [X_BL, Y_BL], [X_BR, Y_BR]]


def brownPoints(center, width, fi):
    """ calculates and returns points on the left and right side of center """
    sin_fi_width = math.sin(fi*math.pi/180) * width
    cos_fi_width = math.cos(fi*math.pi/180) * width

    X_L = center[0] - cos_fi_width
    Y_L = center[1] + sin_fi_width

    X_R = center[0] + cos_fi_width
    Y_R = center[1] - sin_fi_width
    
    

    return [[X_L, Y_L], [X_R, Y_R]]


def plotPoints(h, img, center, height, width, fi):
    """ plots points of shape detector """

    s = math.sqrt(width**2 + height**2)
    delta = math.acos(height/s)*180/math.pi

    print("delta:" + str(delta))
    print("\nfi: " + str(fi))


    for point in darkBluePoints(center, height, width, fi):
        cv.circle(img, (int(point[0]), int(point[1])), 0, (255,0,0), 0)
        cv.circle(h, (int(point[0]), int(point[1])), 0, 0, 0)
        # print(point)

    for point in redPoints(center, height, width, s, fi, delta):
        cv.circle(img, (int(point[0]), int(point[1])), 0, (0,0,255), 0)
        cv.circle(h, (int(point[0]), int(point[1])), 0, 0, 0)
        # print(point)
    
    for point in yellowPoints(center, height, width, fi):
        cv.circle(img, (int(point[0]), int(point[1])), 0, (0,128,255), 0)
        cv.circle(h, (int(point[0]), int(point[1])), 0, 0, 0)
        # print(point)
    
    for point in pinkPoints(center, height, width, fi):
        cv.circle(img, (int(point[0]), int(point[1])), 0, (192,128,255), 0)
        cv.circle(h, (int(point[0]), int(point[1])), 0, 0, 0)
        # print(point)

    for point in brownPoints(center, width, fi):
        cv.circle(img, (int(point[0]), int(point[1])), 0, (0,63,127), 0)
        cv.circle(h, (int(point[0]), int(point[1])), 0, 0, 0)
        # print(point)

    cv.line(img, (center[0], center[1]), (center[0] + int(s * math.sin(fi*math.pi/180)), center[1] + int(s * math.cos(fi*math.pi/180))),(255,0,0))
    
    return h, img

