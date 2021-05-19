import cv2
import numpy as np


def getHalf(image, half):

    """ 
    returns only left or right side of image, also converts and returns tresholded hsv
    input: image
    return: cutImage

    """


    # NOTE: when arm comes into frame, it prevents contour detection and
    # breaks the program

    height, width, channels = image.shape

    if(half == "left"):
        print("left")
        image = image[int(height/20):int(height*19/20),int(width/12):int(width*6/13),:]
    if(half == "right"):
        image = image[int(height*1/20):int(height*19/20),int(width*7/13):int(width*12/13),:]

      

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = cv2.GaussianBlur(hsv[:, :, 0], (7, 7), 15)  # convert to hsv and blur

    ret, h_t = cv2.threshold(h, 60,255, cv2.THRESH_BINARY)
 
    return image, h_t



def getROI(image, pointLeft, pointRight):
    image = image[int(pointLeft[1]):int(pointRight[1]), int(pointLeft[0]):int(pointRight[0])]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = cv2.GaussianBlur(hsv[:, :, 0], (7, 7), 15)  # convert to hsv and blur
    # h = hsv[:, :, 0]

    ret, h_t = cv2.threshold(h, 60,255, cv2.THRESH_BINARY)
 
    return image, h_t



def cutImage(image):
    """
        function cuts image into dimension aaa x bbb, so that only board is in the frame
        input: image
        return: cutImage
    """
    # NOTE: when arm comes into frame, it prevents contour detection and
    # breaks the program

    tresh = 100

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = cv2.GaussianBlur(hsv[:, :, 0], (9, 9), 10)  # convert to hsv and blur

    hsv[hsv < tresh] = 0  # treshold hsv
    hsv[hsv >= tresh] = 360

    # find contours
    contours, hierarchy = cv2.findContours(
        hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    # sort contours, biggest detected is hole frame, second biggest should be
    # the board
    contour_sizes = []

    for i in range(len(contours)):
        contour_sizes.append([cv2.contourArea(contours[i]), i])

    # take second element for sort
    def keyFirst(elem):
        return elem[0]

    # sort list with key
    contour_sizes.sort(key=keyFirst, reverse=True)

    # take second biggest
    cnt = contours[contour_sizes[1][1]]

    # convert to rectangle
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # detect which point is in which corner
    height = image.shape[0]
    width = image.shape[1]

    zgornjiLevo = []
    zgornjiDesno = []
    spodnjiLevo = []
    spodnjiDesno = []

    for element in approx:
        if(element[0, 0] < width / 2):
            if(element[0, 1] < height / 2):
                zgornjiLevo = element[0]
            else:
                spodnjiLevo = element[0]
        else:
            if(element[0, 1] < height / 2):
                zgornjiDesno = element[0]
            else:
                spodnjiDesno = element[0]

    # draw rectangle and cut image
    cv2.rectangle(image, (zgornjiLevo[0], zgornjiLevo[1]),
                  (spodnjiDesno[0], spodnjiDesno[1]), (150, 60, 0), 20)  # draw
    cutImage = image[zgornjiLevo[1]: spodnjiDesno[1],
                     zgornjiLevo[0]: spodnjiDesno[0]]  # cut

    return cutImage

