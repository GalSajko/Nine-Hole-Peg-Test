from shapely.geometry import LineString
from points import redPoints
from rectanglesIntersect import chechIntersection

def getListOfAllowed (pins, height, width, s, delta):
    """
        create list of allowed pins for each pin, so that we know which combos are allowed
        example: if pin 1 overlaps with pin 4 , pin 4 is not allowed

        return dictOfAllowed
    """

    dictOfAllowed = {}

    for chosenPin in pins:
        dictOfAllowed[chosenPin[3]] = []
        for pin in pins:
            if(pin != chosenPin):
                if(not chechIntersection(chosenPin, pin, height, width, s, delta)):
                    dictOfAllowed[chosenPin[3]].append(pin[3])
        print("\n" + str(chosenPin[3]) + "\n")
        
        print((dictOfAllowed[chosenPin[3]]))


    return dictOfAllowed

def getListOfCovering (pins, height, width, s, delta):
    """
        create list of covered pins for each pin, so that we know which combos are covered
        example: if pin 1 overlaps with pin 4 , pin 4 is covered

        return dict of Covered
    """

    dictCovered = {}

    for chosenPin in pins:
        dictCovered[chosenPin[3]] = []
        for pin in pins:
            if(pin != chosenPin):
                if(chechIntersection(chosenPin, pin, height, width, s, delta)):
                    dictCovered[chosenPin[3]].append(pin[3])
        print("\n" + str(chosenPin[3]) + "\n")
        
        print((dictCovered[chosenPin[3]]))


    return dictCovered



def getNumofCovered (pin0, pins, height, width, s, delta):
    """
       returns number of pins that overlap with pin
    """

    num = 0

    for pin in pins:
        if(pin != pin0):
            if(chechIntersection(pin0, pin, height, width, s, delta)):
                num += 1
 
    return num
        
        


