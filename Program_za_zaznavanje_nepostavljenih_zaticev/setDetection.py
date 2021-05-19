import numpy as np
import cv2 as cv
import time
from tqdm import tqdm
import math
import random
from shapely.geometry import LineString
import sys
from tempfile import TemporaryFile



from imagePrepare import getHalf, cutImage
from findPossiblePins import *
from drawRect import *
from rectanglesIntersect import *
from pinCombos import *
from getOriginals import *
from detectionFun import *


def saveCoord(event, x, y, flags, param):
    """
        saves coord on click event
    """
    
    global leftCoord
    global rightCoord

    if(event == 1):
        print("Left side:")
        print(x)
        print(y)
        leftCoord = [x,y]
          
    if(event == 2):
        print("Right side:")
        print(x)
        print(y)
        rightCoord = [x,y]

    image = clone.copy()
    cv.rectangle(image, (leftCoord[0], leftCoord[1]), (rightCoord[0], rightCoord[1]), (144,44,55), 1)
    cv2.imshow("image", image)
      
       

"""
1. load image
2. click left 
3. click right
4. input num of pins
5. auto set constants until correct num of pins are found
"""

# accept arguments and prepare data
imageName = sys.argv[1]
numOfPins = sys.argv[2]
fullName = '../DEMO/' + imageName

image =  cv.imread(fullName)
clone = image.copy()

leftCoord = [0,0]
rightCoord = [0,0]


# set area of pins

print("Set pins area: ")
cv.namedWindow("image")
cv.setMouseCallback("image", saveCoord)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save coord to files

np.save('leftCoordFile', leftCoord)
np.save('rightCoordFile', rightCoord)    


# set size of pins
try:
    height = np.load('width')
    width = np.load('width')
except FileNotFoundError :
    height = 12.5*1.5
    width = 3*1.5
    print('np width or height not found')


correctInARow = 0

# while we dont get 5 correct in a row detections, we keep fixing parameters

while(correctInARow < 5):

    numOfDet = detectionFun(fullName, height, width, leftCoord, rightCoord, "left", False, 1)

    if (int(numOfDet) == int(numOfPins)):
        correctInARow += 1
    else:
        correctInARow = 0

        if(int(numOfDet) > int(numOfPins)):
            height = height*1.05
            width = width*1.05
        
        if(int(numOfDet) < int(numOfPins)):
            height = height*0.95
            width = width*0.95
    
    if(correctInARow == 5):
        break

    print(height)
    print(width)

np.save('height', height)
np.save('width', width)

detectionFun(fullName, height, width, leftCoord, rightCoord, "left", True, 1)


        










