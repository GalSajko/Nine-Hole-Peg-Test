from rectanglesIntersect import *
from pinCombos import  getNumofCovered
from tqdm import tqdm

def lessCovered(pins, height, width, s, delta):
    """
        removes one of covering pins

        return: array that has one less element than inputed
         
    """

    addressesBook = [] # list of addresses of all pins
    adressesRemoved = [] # list of removed addresses
    numopa = 0

    pins.reverse()

    for pin in pins:
        addressesBook.append(pin[3])

    for address0 in tqdm(addressesBook):
        if(not address0 in adressesRemoved): # if address is already removed ... skip it
            element0 = next(filter(lambda pin: pin[3] == address0, pins)) # get value of element
            
            for address1 in addressesBook:
                if(address0 != address1): # if we have same address twice ... skip it
                    if(not address1 in adressesRemoved): # if address is already removed ... skip it
                        element1 = next(filter(lambda pin: pin[3] == address1, pins)) # get value of element
                        if(chechIntersection(element0, element1, height, width, s, delta)): # if two pins intersect
                            
                            # check which two of two pins has more intersecting pins and remove it
                            cov0 = getNumofCovered(element0, pins, height, width, s, delta)
                            cov1 = getNumofCovered(element1, pins, height, width, s, delta)


                            if(cov0 > cov1):
                                pins.remove(element0)
                                adressesRemoved.append(address0)
                                break
                            
                            if(cov0 == cov1 and cov0 != 0):
                                pins.remove(element0)
                                adressesRemoved.append(address0)
                                break
                            
                            if(cov0 < cov1):
                                pins.remove(element1)
                                adressesRemoved.append(address1)
                        
    print(len(pins))

    return pins
                        # print((element1))
                                                
                    
                   


                        




    
            

        
            

