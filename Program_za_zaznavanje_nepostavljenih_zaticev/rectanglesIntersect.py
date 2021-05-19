from shapely.geometry import LineString
from points import redPoints

def chechIntersection (rectangle1, rectangle2, height, width, s, delta):

    """
        checks if two rectangles intersect

        returns:
            TRUE if intersect,
            FALSE if not intersect
    """

    # first get corners of rectangles

    corners1 = redPoints((rectangle1[0][0], rectangle1[0][1]), height, width,s, rectangle1[1], delta)
    corners2 = redPoints((rectangle2[0][0], rectangle2[0][1]), height, width,s, rectangle2[1], delta)

    # generate side lines of rectangles

    top1 = LineString([(corners1[0][0], corners1[0][1]),(corners1[1][0], corners1[1][1])])
    top2 = LineString([(corners2[0][0], corners2[0][1]),(corners2[1][0], corners2[1][1])])

    bot1 = LineString([(corners1[2][0], corners1[2][1]),(corners1[3][0], corners1[3][1])])
    bot2 = LineString([(corners2[2][0], corners2[2][1]),(corners2[3][0], corners2[3][1])])

    left1 = LineString([(corners1[0][0], corners1[0][1]),(corners1[2][0], corners1[2][1])])
    left2 = LineString([(corners2[0][0], corners2[0][1]),(corners2[2][0], corners2[2][1])])

    right1 = LineString([(corners1[1][0], corners1[1][1]),(corners1[3][0], corners1[3][1])])
    right2 = LineString([(corners2[1][0], corners2[1][1]),(corners2[3][0], corners2[3][1])])

    # check if any of the lines intersect
    # if they intersect, rectangles intersect

    status = False
    
    # top1
    if(not status):
        status = top1.intersects(top2)

    if(not status):
        status = top1.intersects(bot2)

    if(not status):
        status = top1.intersects(left2)

    if(not status):
        status = top1.intersects(right2)

    
    # bot1
    if(not status):
        status = bot1.intersects(top2)
    
    if(not status):
        status = bot1.intersects(bot2)

    if(not status):
        status = bot1.intersects(left2)

    if(not status):
        status = bot1.intersects(right2)

    # left1
    if(not status):
        status = left1.intersects(top2)

    if(not status):
        status = left1.intersects(left2)

    if(not status):
        status = left1.intersects(right2)

    if(not status):
        status = left1.intersects(bot2)

    # right1
    if(not status):
        status = right1.intersects(top2)

    if(not status):
        status = right1.intersects(left2)

    if(not status):
        status = right1.intersects(right2)

    if(not status):
        status = right1.intersects(bot2)

    return status

    