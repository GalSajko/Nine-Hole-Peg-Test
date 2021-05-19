from pickle import TRUE
import numpy as np
import cv2 as cv
import time
from tqdm import tqdm
import math
import random
from shapely.geometry import LineString


from imagePrepare import getHalf, cutImage
from findPossiblePins import *
from drawRect import *
from rectanglesIntersect import *
from pinCombos import *
from getOriginals import *
from detectionFun import *

#    height = 12.5
#     width = 3

leftPoint = [25, 25]
rightPoint = [280, 280]

detectionFun("v1/WIN_20210401_12_22_06_Pro.jpg", 12.5, 3, leftPoint, rightPoint, "left", True, 2)
detectionFun("v1/WIN_20210401_12_22_01_Pro.jpg", 12.5, 3, leftPoint, rightPoint, "left", True, 2)
detectionFun("v1/WIN_20210401_12_22_10_Pro.jpg", 12.5, 3, leftPoint, rightPoint, "left", True, 2)
detectionFun("v1/WIN_20210401_12_22_20_Pro.jpg", 12.5, 3, leftPoint, rightPoint, "left", True, 4)






























