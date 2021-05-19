import math
from tqdm import tqdm
from imagePrepare import *
from points import *
import itertools
from drawRect import *

def checkPoints(h, points):
    """
     checks if all given points are in the h (hsv) image
     returns TRUE / FALSE 
    """

    # check for every point if it is inside
    for point in points:
        if(h[int(point[1])][int(point[0])] < 200):
            return False
    return True



def findPinsLeft(h, height, width):
    """ 
        find possible pins, that are still in holding area 

        return: list([center, fi])
    """

    s = math.sqrt(width**2 + height**2)
    delta = math.acos(height/s)*180/math.pi

    print("\ndelta: " + str(delta))
    

    s_height, s_width = h.shape

    possiblePins = [] # list of all possible pins positions

    # go over all points in image
    for y_num, y in tqdm(enumerate(h), total=s_height):
        for x_num, x in enumerate(y):
            
            if(h[y_num, x_num] > 50):  # check if center point is black ... if it is, than it is on pin

                for fi in range(-90,90,5): # for each point check all possible rotations ( but not "mirrored" )
                    status = True # if all tests check out, point is center of a pin

                    cornerPoints = redPoints([x_num, y_num], height, width,s, fi, delta)

                    # check if points are inside the image
                    for point in cornerPoints:
                        if(point[0] < 0 or point[1] < 0 ): 
                            status = False
                        if(point[0] > s_width-1 or point[1] > s_height-1): 
                            status = False
                    
                    # go thrue a list of checks of all points
                    center = [x_num, y_num]

                    if(status):
                        status = checkPoints(h, [center])
                    
                    if(status):
                        status = checkPoints(h, cornerPoints)
                    
                    if(status):
                        status = checkPoints(h, darkBluePoints(center, height, width, fi))
                    
                    if(status):
                        status = checkPoints(h, yellowPoints(center, height, width, fi))
                    
                    if(status):
                        status = checkPoints(h, pinkPoints(center, height, width, fi))

                    if(status):
                        status = checkPoints(h, brownPoints(center, width, fi))

                    # if all points are in the shape, save center point and fi as possible pin position

                    if(status):
                        possiblePins.append([center, fi])

    return possiblePins
                    
                    

def filteredCandidates ( possiblePins, radious, fiErr):
    """
        checks possiblePins, returns only a few most likely locations of pins
        for each pin location iterates over every pin and sums number of pins in the same direction
    
        return: list([center, fi, sumOfpins, pinNum])

        sumOfpins - number of similarily layed pins in proximity

        pinNum - pin designation number
    """
    allPins = [] # possiblePins with added number of close pins

    radiousSq = radious**2

    for pinOfInt in tqdm(possiblePins):

        sumOfPins = 0
        # iterate over every other pin location and return number of close pins looking in the same direction
        for pin in possiblePins:
            if(abs(pin[1]-pinOfInt[1]) < fiErr): # check if pins are alligned
                if((pin[0][0] - pinOfInt[0][0])**2 + (pin[0][1] - pinOfInt[0][1])**2 < radiousSq): # check if pins centers are close
                    sumOfPins += 1
        
        allPins.append([pinOfInt[0], pinOfInt[1], sumOfPins])
    
    filtered = []

    # sort pins from one with most relatives in proximity to one with least
    allPins.sort(key = lambda x : x[2], reverse = True)   

    finalPins = []
    counter0 = 0
    # only keep pins with most relatives
    for pinOfInt in allPins:
        exists = False
        for pin in (finalPins):
            if(abs(pin[1]-pinOfInt[1]) < fiErr and (pin[0][0] - pinOfInt[0][0])**2 + (pin[0][1] - pinOfInt[0][1])**2 < radiousSq): # check if pins are alligned and check if pins centers are close
                exists = True
                      
        if(not exists):
            
            pinOfInt.append(counter0)
            counter0 += 1
            
            finalPins.append(pinOfInt)

    print("\nAll pins: " + str(len(possiblePins)))
    print("\nPins left:  " + str(len(finalPins)) + "\n")

    return finalPins


