import numpy as np
import cv2 as cv
import time
from tqdm import tqdm
import math
import random
from shapely.geometry import LineString


from imagePrepare import getHalf, cutImage, getROI
from findPossiblePins import *
from drawRect import *
from rectanglesIntersect import *
from pinCombos import *
from getOriginals import *
# from detectionFun import *


def detectionFun (imageName, height, width, leftPoint, rightPoint, hand, draw, repeats = 1):

    """
        runs all steps of detection

        return num of detected pins 
    """
    resizeFactor = 4

    t = time.time()
    #   ---------------------------------         
    #   ---------------------------------       

    fullName = '../DEMO/'+imageName
    demoImage0 = cv.imread(fullName)

    image = cutImage(demoImage0)

    img, h = getHalf(image, hand)

    # img, h = getROI(demoImage0, leftPoint, rightPoint)



    if(draw):
        cv2.imshow("demo0", h)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    s_height, s_width = h.shape
    h = cv.resize(h, (int(s_height/resizeFactor), int(s_width/resizeFactor)))
    img = cv.resize(img, (int(s_height/resizeFactor), int(s_width/resizeFactor)))


    possiblePins = findPinsLeft(h, height, width)

    possiblePins = filteredCandidates(possiblePins, 2, 2)

    s = math.sqrt(width**2 + height**2)
    delta = math.acos(height/s)*180/math.pi 

    # run detection multiple times, so that we get optimal configuration of detected pins
    results = []

    for i in range(repeats):

        pins2 = possiblePins.copy()

        random.shuffle(pins2)

        pins = lessCovered(pins2, height, width, s, delta)

        results.append([len(pins), pins])

    # sort results, take out one with max number of detected pins
    results = sorted(results,key=lambda x: x[0], reverse=True)
    pins = results[0][1] # best run / most recognized
    # show number of recognized pins
    print("\n\nRESULTS:")

    for elem in results:
        print(elem)




    # display execution times
    print("\n---\nExecution time [s]: " + str( time.time()-t))

    # display image with recoginzed pins
    if(draw):

        for pin in pins:
            
            cv.circle(img, (int(pin[0][0]), int(pin[0][1])), 0, (0,0,255))
            # print(pin)
            # points = redPoints((pin[0][0], pin[0][1]), height, width, s, pin[1], delta)
            color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
            filledRect(h, img, pin, width, height, color)

        cv.namedWindow("h", cv.WINDOW_NORMAL) 
        cv.namedWindow("image", cv.WINDOW_NORMAL) 
        cv.imshow('h',h)
        cv.imshow('image',img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    return len(pins)

if __name__ == "__main__":
        
    leng = detectionFun('v1/WIN_20210401_12_22_06_Pro.jpg', 12.5, 3, "left", False, 2)
    

#    height = 12.5
#     width = 3