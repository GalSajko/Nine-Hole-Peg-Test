import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm

from helperMethods import *
from points import *


def findCircles(blurImg, tH):
    # Find candidates for setted pins.
    pinsCandidates = cv.HoughCircles(
        blurImg,
        cv.HOUGH_GRADIENT,
        3,
        50,
        param1=tH,
        param2=5,
        minRadius=3,
        maxRadius=10)
    # Find collection area for unsetted pins.
    collectionArea = cv.HoughCircles(
        blurImg,
        cv.HOUGH_GRADIENT,
        3,
        50,
        param1=tH,
        param2=100,
        minRadius=185,
        maxRadius=215)
    
    return pinsCandidates, collectionArea


def detectHand(currentSum, startSum):
    if (int(startSum) - int(currentSum) > 5000000):
        return True
    return False


# def findRectangles(image, circle):
#     t = time.time() # algoritem execution time
    
#     # cut out only area of interest
#     x1 = int(circle[0] - circle[2] - 30) 
#     x2 = int(circle[0] + circle[2] + 30)
#     y1 = int(circle[1] - circle[2] - 30)
#     y2 = int(circle[1] + circle[2] + 30)

#     image = image[y1:y2,x1:x2]
    
#     # convert to hsv
#     hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv_image)

#     # convert to gray
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
               
#     # blur       
#     h = cv.GaussianBlur(h, (19,19), 3)

#     # treshold
#     h[h > 50] = 255

#     # detect edges
#     edges = cv.Canny(h,50,100)

#     # find contours
#     contours, hierarchy =  cv.findContours(edges, 1,2)

#     # method 1 of finding number of pins left, calculating total contour area
#     #  and deviding by approximate area of one pin
#     area = 0

    
    
#     # array of potential lengths, that are long enough to be a pin
#     potential_lengths = []

#     for num, cnt in enumerate(contours):

#         # we use every second contour, as they are for some reason drawn twice almost same
#         # HACK: to fix later

#         # min_len     
#         potential_lengths.append([])

#         if num%2 == 0:

#             cnt = cv.approxPolyDP(cnt, 2, True)
#             area  += cv.contourArea(cnt)
#             cv.drawContours(image, [cnt], -1, (255,50,25), 3)
            
#             # we look at the distance between all points in the contour
#             # we create vectors: (point1, point2, distanceSquared)

#             for point in cnt:
#                 # print("point : " + str(point[0]))
#                 cv.circle(image, (point[0,0], point[0,1]),1,(0,0,255))
                
#                 for point2 in cnt:
#                     if(point[0] != point2[0]).all():
                        
#                         # if not same point
#                         distanceSquared = (point[0,0]-point2[0,0])**2 + (point[0,1]-point2[0,1])**2
#                         if(distanceSquared > 10500 and distanceSquared < 11000):
#                             potential_lengths[num].append([point, point2, distanceSquared])
#                         # print("P1: " + str(point[0]) + "  P2: " + str(point2[0]) + "  Distance: " + str(distanceSquared))
                        
#         for line in potential_lengths[num]:
#             cv.line(image, (line[0][0][0], line[0][0][1]),(line[1][0][0], line[1][0][1]), (0,255,0), 1)
#             # print(line[1][0][0])

        
#     # for element in potential_lengths[2]:
#     #     print(element)
            

#     print("Št.pinov glede na površino:  " +  str(int(area / 3200)))
   
    


#     print("\n---\nExecution time [s]: " + str( time.time()-t))
#     cv.imshow('image',image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def findRectangles2(image, circle):
#     t = time.time() # algoritem execution time
    
#     # cut out only area of interest
#     x1 = int(circle[0] - circle[2] - 30) 
#     x2 = int(circle[0] + circle[2] + 30)
#     y1 = int(circle[1] - circle[2] - 30)
#     y2 = int(circle[1] + circle[2] + 30)

#     image = image[y1:y2,x1:x2]
    
#     height, width, channels = image.shape

#     image_small = cv.resize(image,(int(width/2), int(height/2)) )
    
#     # convert to hsv
#     hsv_image = cv.cvtColor(image_small, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv_image)
               
#     # blur       
#     # h = cv.GaussianBlur(h, (19,19), 3)

#     # treshold
#     h[h > 50] = 255

#     # convert image to smaller one, for faster computations

#     # h_small = cv.resize(h,(int(width/6), int(height/6)) )
#     h[h > 100] = 255
#     h[h <= 100] = 0

#     # detect edges
#     edges = cv.Canny(h,50,100)

#     # find contours
#     contours, hierarchy =  cv.findContours(edges, 1,2)

#     cv.drawContours(image, contours, -1, (0,255,0), 3)

#     half_length = 20 # half of length of peg on its side
#     quarter_length = int(half_length/2)
#     half_width = 10
    
#     s_height, s_width = h.shape
#     both_in = 0

#     # print(h.shape)
 
#     # go over all elements
#     for y_num, y in enumerate(h):
#         for x_num, x in enumerate(y):
#              # check if center point is black ... if it is, than it is on pin
#             if(h[y_num, x_num] < 50):
#                 for fi in range(-90,0,5):
#                     cos_fi = (math.cos(fi))
#                     sin_fi = (math.sin(fi))
#                     # print((sin_fi))

#                     # calculate right center point
                    
#                     xc_r = x_num + int(half_width * sin_fi)
#                     yc_r = y_num + int(half_width * cos_fi)
#                     if(xc_r < 0 or yc_r < 0): 
#                             break
#                     if(xc_r > s_width-1 or yc_r > s_height-1): 
#                         break

#                     # check if right center point is black ... if it is, than it is on pin
#                     if(h[yc_r, xc_r] < 50):
#                         # calculate left border upper and lower point
#                         # (top left if 0,0)
#                         upper_x = x_num - int(half_length * cos_fi)
#                         upper_y = y_num - int(half_length * sin_fi)
#                         lower_x = x_num + int(half_length * cos_fi)
#                         lower_y = y_num + int(half_length * sin_fi)

                        
#                         # remove points that are out of frame
#                         if(upper_x < 0 or upper_y < 0 or lower_x < 0 or lower_y < 0): 
#                             break
#                         if(upper_x > s_width-1 or upper_y > s_height-1 or lower_x > s_width-1 or lower_y > s_height-1): 
#                             break
                        

                    
#                         # check if upper point is black 
#                         if(h[upper_y, upper_x] < 50):
#                             #check if upper right is black
#                                 # calculate right border points upper and lower point
#                                 upper_x_r = upper_x + int(half_width * sin_fi)
#                                 upper_y_r = upper_y + int(half_width * cos_fi)
#                                   # remove points that are out of frame
#                                 if(upper_x_r < 0 or upper_y_r < 0): 
#                                     break
#                                 if(upper_x_r > s_width-1 or upper_y_r > s_height-1): 
#                                     break
#                                 # check if upper right is black
#                                 if(h[upper_y_r, upper_x_r] < 50):
#                                     lower_x_r = lower_x + int(half_width * sin_fi)
#                                     lower_y_r = lower_y + int(half_width * cos_fi)
#                                       # remove points that are out of frame
#                                     if( lower_x_r < 0 or lower_y_r < 0): 
#                                         break
#                                     if(lower_x_r > s_width-1 or lower_y_r > s_height-1): 
#                                         break
#                                     # check if lower point is black
#                                     if(h[lower_y, lower_x] < 50):
#                                         # check if lower right is black
#                                         if(h[lower_y_r, lower_x_r] < 50):
#                                             # calculate two more points, on quarters o--o--X--o--o
#                                             q_upper_x = x_num - int(quarter_length * cos_fi)
#                                             q_upper_y = y_num - int(quarter_length * sin_fi)
#                                             q_lower_x = x_num + int(quarter_length * cos_fi)
#                                             q_lower_y = y_num + int(quarter_length * sin_fi)
                                            
#                                             # check if upper quarter point is in
#                                             if(h[q_upper_y, q_upper_x] < 50):
#                                                 # check if lower quarter point is in
#                                                 if(h[q_lower_y, q_lower_x] < 50):
#                                                     # calculate two more points right, on quarters o--o--X--o--o
#                                                     q_upper_x_r = q_upper_x + int(half_width * sin_fi)
#                                                     q_upper_y_r = q_upper_y + int(half_width * cos_fi)
#                                                     q_lower_x_r = q_lower_x + int(half_width * sin_fi)
#                                                     q_lower_y_r = q_lower_y + int(half_width * cos_fi)
                                                    
#                                                     # check if upper quarter point is in
#                                                     if(h[q_upper_y_r, q_upper_x_r] < 50):
#                                                         # check if lower quarter point is in
#                                                         if(h[q_lower_y_r, q_lower_x_r] < 50):
                                                            
#                                                             cv.line(image_small, (lower_x, lower_y), (upper_x, upper_y),(0,0,255), 1)
#                                                             # cv.line(image_small, (lower_x_r, lower_y_r), (upper_x_r, upper_y_r),(255,0,0), 1)
#                                                             both_in += 1
                                                

#     print(both_in)
    
                      

# def findRectangles3(image, circle):
#     t = time.time() # algoritem execution time
    
#     # cut out only area of interest
#     x1 = int(circle[0] - circle[2] - 30) 
#     x2 = int(circle[0] + circle[2] + 30)
#     y1 = int(circle[1] - circle[2] - 30)
#     y2 = int(circle[1] + circle[2] + 30)

#     image = image[y1:y2,x1:x2]
    
#     height, width, channels = image.shape

#     image_small = cv.resize(image,(int(width/2), int(height/2)) )
    
#     # convert to hsv
#     hsv_image = cv.cvtColor(image_small, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv_image)
               
#     # blur       
#     # h = cv.GaussianBlur(h, (19,19), 3)

#     # treshold
#     h[h > 50] = 255

#     # convert image to smaller one, for faster computations

#     # h_small = cv.resize(h,(int(width/6), int(height/6)) )
#     h[h > 100] = 255
#     h[h <= 100] = 0

#     # detect edges
#     edges = cv.Canny(h,50,100)

#     # find contours
#     contours, hierarchy =  cv.findContours(edges, 1,2)

#     cv.drawContours(image, contours, -1, (0,255,0), 3)

#     half_length = 30 # half of length of peg on its side
#     quarter_length = int(half_length/2)
#     octet_length = int(half_length/4)
#     half_width = 10
    
#     s_height, s_width = h.shape
#     both_in = 0
#     lines = []

#     # print(h.shape)
 
#     # go over all elements
#     for y_num, y in enumerate(h):
#         for x_num, x in enumerate(y):
#              # check if center point is black ... if it is, than it is on pin
#             if(h[y_num, x_num] < 50):
#                 for fi in range(-90,90,20):
#                     cos_fi = (math.cos(fi))
#                     sin_fi = (math.sin(fi))
                    
#                     # calculate left border upper and lower point
#                     # (top left if 0,0)
#                     upper_x = x_num - int(half_length * cos_fi)
#                     upper_y = y_num - int(half_length * sin_fi)
#                     lower_x = x_num + int(half_length * cos_fi)
#                     lower_y = y_num + int(half_length * sin_fi)

                    
#                     # remove points that are out of frame
#                     if(upper_x < 0 or upper_y < 0 or lower_x < 0 or lower_y < 0): 
#                         break
#                     if(upper_x > s_width-1 or upper_y > s_height-1 or lower_x > s_width-1 or lower_y > s_height-1): 
#                         break
                    
#                     # check if upper point is black 
#                     if(h[upper_y, upper_x] < 50):                
#                         # check if lower point is black
#                         if(h[lower_y, lower_x] < 50):
#                             # calculate two more points, on quarters o--o--X--o--o
#                             q_upper_x = x_num - int(quarter_length * cos_fi)
#                             q_upper_y = y_num - int(quarter_length * sin_fi)
                            
                            
#                             # check if upper quarter point is in
#                             if(h[q_upper_y, q_upper_x] < 50):
#                                 # check if lower quarter point is in
#                                 q_lower_x = x_num + int(quarter_length * cos_fi)
#                                 q_lower_y = y_num + int(quarter_length * sin_fi)
#                                 if(h[q_lower_y, q_lower_x] < 50):
#                                     # two more points - eights
#                                     o_upper_x = x_num - int(octet_length * cos_fi)
#                                     o_upper_y = y_num - int(octet_length * sin_fi)

#                                     if(h[o_upper_y, o_upper_x] < 50):
#                                         o_lower_x = x_num + int(octet_length * cos_fi)
#                                         o_lower_y = y_num + int(octet_length * sin_fi)
                                        
#                                         if(h[o_lower_y, o_lower_x] < 50):
                                                                               
#                                             # cv.line(image_small, (lower_x, lower_y), (upper_x, upper_y),(0,0,255), 1)
#                                             # cv.line(image_small, (lower_x_r, lower_y_r), (upper_x_r, upper_y_r),(255,0,0), 1)
#                                             both_in += 1
#                                             lines.append([x_num,y_num, fi])

#     candidates = []
#     better_candidates = []
#     best_candidates = []
#     radius = 25

#     # for every line in array that we got check if in proximity to its center point are other lines with same fi angle
#     for line in lines:
#         neighbours = 0
#         for element in lines:
#             # check if lines are under the same angle
#             if((element[2] == line[2])):
#                 distance_s = (line[0]-element[0])**2 + (line[1]-element[1])**2
#                 if (distance_s < radius):
#                     neighbours += 1
#         if(neighbours > 30):
#             cv.circle(image_small, (line[0], line[1]),1,(0,255,0),1)
#             better_candidates.append([line[0], line[1], line[2], False])
#             # print(neighbours)
#     print(len(better_candidates))
#     print(image_small.shape)
    
#     for candidate in better_candidates:
#         close = 0
        
#         if(not candidate[3]):
#             best_candidates.append(candidate)
#             candidate[3] = True
#             for element in better_candidates:
                
#                 distance_s = (candidate[0]-element[0])**2 + (candidate[1]-element[1])**2
#                 angle = abs(element[2] - candidate[2])
                
#                 if(distance_s < 1000 and angle < 30 and  not(element[3])): # nastavi parametre
#                     element[3] = True
#             cv.circle(image_small, (candidate[0], candidate[1]), 1, (255,0,0))
#             pt1 = (candidate[0] - int(half_length * math.cos(candidate[2])), candidate[1] - int(half_length * math.sin(candidate[2])))
#             pt2 = (candidate[0] + int(half_length * math.cos(candidate[2])), candidate[1] + int(half_length * math.sin(candidate[2])))
#             cv.line(image_small, pt1, pt2,(0,0,255))
#     print((best_candidates))
#     print(len(best_candidates))

        
#     print(both_in)
    

     

#     print("\n---\nExecution time [s]: " + str( time.time()-t))
#     # cv.namedWindow("h", cv.WINDOW_NORMAL) 
#     cv.namedWindow("image_small", cv.WINDOW_NORMAL) 
#     # cv.imshow('h',h)
#     cv.imshow("image", image)
#     cv.imshow('image_small', image_small)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    
    
#     # TODO:
#     # -> create a better rectangle detection, with more points and width.
#     # -> lower image resolution a bit
#     # -> find how to calculate area of rectangles overlapping





# def findRectangles4(image, circle):
#     t = time.time() # algoritem execution time
    
#     # cut out only area of interest
#     x1 = int(circle[0] - circle[2] - 30) 
#     x2 = int(circle[0] + circle[2] + 30)
#     y1 = int(circle[1] - circle[2] - 30)
#     y2 = int(circle[1] + circle[2] + 30)

#     image = image[y1:y2,x1:x2]
    
#     height, width, channels = image.shape

#     image_small = cv.resize(image,(int(width/2), int(height/2)) )
    
#     # convert to hsv
#     hsv_image = cv.cvtColor(image_small, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv_image)
               
#     # blur       
#     # h = cv.GaussianBlur(h, (19,19), 3)

#     # treshold
#     h[h > 50] = 255

#     # convert image to smaller one, for faster computations

#     # h_small = cv.resize(h,(int(width/6), int(height/6)) )
#     h[h > 100] = 255
#     h[h <= 100] = 0

#     # detect edges
#     edges = cv.Canny(h,50,100)

#     # find contours
#     contours, hierarchy =  cv.findContours(edges, 1,2)

#     cv.drawContours(image_small, contours, -1, (0,255,0), 3)

#     width = 3
#     height = 25
    

#     s_height, s_width = h.shape

#     fi = 45
    
#     plotPoints(image_small, [168, 85], height, width, fi)
        

#     print("\n---\nExecution time [s]: " + str( time.time()-t))
#     cv.namedWindow("h", cv.WINDOW_NORMAL) 
#     cv.namedWindow("image_small", cv.WINDOW_NORMAL) 
#     cv.imshow('h',h)
#     cv.imshow("image", image)
#     cv.imshow('image_small', image_small)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    
    
#     # TODO:
#     # -> create a better rectangle detection, with more points and width.
#     # -> lower image resolution a bit
#     # -> find how to calculate area of rectangles overlapping

