#!/usr/bin/env python

""" Image Detection Module.
    in: video file format avi or image sequence
    output: data file with angle position of detection according params
"""
__author__ =    "Alejandro Pequeno"
__copyright__ = "Copyright 2017"
__license__ =   "GPL"
__email__ =     "alpeq16@student.sdu.dk"

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import time

"""         Global Parameters        """
# Visual Mode
DEBUG = 1
# Video Characteristics
#   Params Static Case
Minline_length_static = 50
Maxline_gap_static = 0
Threshold_static = 50
# Point Extraction Variables
LIMITSEP = 50       # Minimal separation between points Isolation point function
MaxVar = 200        # Maximal distance allowed between frames 200
# LED Detection
SizeWin = 200       # Length in pixels of window around the led in order to Detect
LedStart = 0        # State of Led at the beginning
LedThreshold = 15000
# File Directories
Current_folder = os.path.dirname(os.path.abspath(__file__))
Input_directory = os.path.join(Current_folder, 'src')
Output_directory = os.path.join(Current_folder, 'out')

"""                                   """



def main():
    # Graphical mode activation
    if DEBUG:
        cv2.namedWindow("BE", cv2.WINDOW_NORMAL)
        cv2.namedWindow("LD", cv2.WINDOW_NORMAL)
    # Batch Tracking, all files in the specified directory
    allfiles = [f for f in os.listdir(Input_directory) if os.path.isfile(os.path.join(Input_directory, f))]
    init_points = list()
    distal_points = list()
    l_points = list()
    filter_points = list()
    for a in allfiles:
        in_name = os.path.join(Input_directory, a)
        # """ Selecting points from the image """
        cap = cv2.VideoCapture(in_name)
        ret, frame = cap.read()  # Initialize Stream
        fig, ax = plt.subplots()
        #   Initial
        ax.set(title='Click in Antenna INITIAL points Left, Right')
        plt.imshow(frame)
        (ant_iniL, ant_iniR) = plt.ginput(2)
        ant_iniL = (int(ant_iniL[0]), int(ant_iniL[1]))
        ant_iniR = (int(ant_iniR[0]), int(ant_iniR[1]))
        #   Distal
        ax.set(title='Click in Antenna DISTAL points Left, Right')
        plt.imshow(frame)
        (end_pL, end_pR) = plt.ginput(2)
        end_pL = (int(end_pL[0]), int(end_pL[1]))
        end_pR = (int(end_pR[0]), int(end_pR[1]))
        # LED
        ax.set(title='Click in LED position')
        plt.imshow(frame)
        (point) = plt.ginput(1)
        point = (int(point[0][0]), int(point[0][1]))


        # Filter
        ax.set(title='Click in Filter (UpperLeft and UpperRight of a square area)')
        plt.imshow(frame)
        (filterL,filterR) = plt.ginput(2)
        filterL = (int(filterL[0]), int(filterL[1]))
        filterR = (int(filterR[0]), int(filterR[1]))
        plt.close()

        # Add initial points
        l_points.append(point)
        init_points.append((ant_iniL, ant_iniR))
        distal_points.append((end_pL, end_pR))
        filter_points.append((filterL,filterR))

    for a in allfiles:
        in_name = os.path.join(Input_directory, a)
        out_name = os.path.join(Output_directory, a.split('.')[0] + '.txt')
        trackVideo(in_name, out_name, init_points.pop(0), distal_points.pop(0), l_points.pop(0),filter_points.pop(0))

    return


def trackVideo(input_name, output_name, init_points, distal_points, point, filter_points):
    """
        Main function of tracking
        Inputs a video file
        Outputs a sequence of positional points correlated by the lines
    """
    print "Reading file " + input_name
    # """ Initialize variables """
    ant_iniL = init_points[0]
    ant_iniR = init_points[1]
    end_pL   = distal_points[0]
    end_pR   = distal_points[1]
    filterL = filter_points[0];
    filterR = filter_points[1];
    count = 0                           # Count loop for file manipulation
    prev_angL = 0                      # Initialize memory previous angle Left
    prev_angR = 0                      # Initialize memory previous angle Right
    led_state = LedStart               # Inital state of the LED
    mem_points = {"Left": end_pL, "Right": end_pR}
    #   Dynamic Background Extraction Properties
    fgbg500 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=True) #100
    cap = cv2.VideoCapture(input_name)    # Video Capture and Windowing
    ret, frame = cap.read()             # Initialize Stream
    #   Open Output points file
    fo = open(output_name, "wb")
    fo.write(" \tSample\t\tLeft\t\t\tRight\t\t\tAngleLeft\tAngleRight\tLedState\n")
    #   Use first image as Background
    fgmask500 = fgbg500.apply(frame)
    fgmask500 = cv2.dilate(fgmask500, kernel=np.ones((5, 5)), iterations=1)  # Clean image extraction
    #   Crop image of the LED and calculate pixel colour
    crop_img = frame[max(0, point[1] - SizeWin / 2):min(point[1] + SizeWin / 2, frame.shape[0]),
               max(0, point[0] - SizeWin / 2):min(point[0] + SizeWin / 2,
                                                  frame.shape[1])]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # Init
    sumPix = sum(cv2.sumElems(crop_img))
    ret, frame = cap.read()    
    t = time.clock()
    # """ Infinite loop until no more images in video buffer """
    while (ret):

        # """ Crop and compare LED Image """
        crop_img = frame[ max(0,point[1] - SizeWin/2):min(point[1] + SizeWin/2 , frame.shape[0] ), max(0, point[0] - SizeWin/2):min(point[0] + SizeWin/2, frame.shape[1] )]  # Crop from x, y, w, h -> 100, 200, 300, 400
        if sum(cv2.sumElems(crop_img)) - sumPix > LedThreshold:
            led_state += 1
        elif sum(cv2.sumElems(crop_img)) - sumPix < -LedThreshold:
            led_state -= 1
        sumPix = sum(cv2.sumElems(crop_img))
        # See Image cropped
        if DEBUG:
            cv2.imshow("LED", crop_img)

        #  """ Move Detection """
        #   Background Extraction (See above for params)
        fgmask500 = fgbg500.apply(frame)
        fgmask500 = cv2.dilate(fgmask500, kernel=np.ones((5, 5)), iterations=1)  # Clean image extraction


        #  """ Line Detection """
        blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        #   Get Lines
        lines = cv2.HoughLinesP(fgmask500, 1, np.pi / 180, Threshold_static, lines=None,
                                minLineLength=Minline_length_static, maxLineGap=Maxline_gap_static)
        #  """ Draw Lines """
        #   Filter lines with points in the squares defined by user
        #   The filter will not be applied if the tracking points obtained previously are closed to the regions.
        try:
            for draw in lines:
                for x1, y1, x2, y2 in draw:
		    # Filter
                    if y1 > min(filterL[1], filterR[1]) and (x1 > filterL[0]) and (x1 < filterR[0]) \
                            and mem_points.get("Left")[1] < 0.9*(min(filterL[1], filterR[1])) and mem_points.get("Right")[1] < 0.9*(min(filterL[1], filterR[1]) ):
                        continue
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        except:
            pass

        # """ Points Extraction """
        detect_points = extractPoints(blank_image, end_pL, end_pR)

        # """ PostProcess Points """
        # Select points in Angle Calculation
        (ant_endL, ant_endR) = detection(detect_points, mem_points)
        mem_points["Left"] = ant_endL
        mem_points["Right"] = ant_endR
        # Calculate Angles
        angleL = get_angle(ant_iniL, ant_endL)
        angleR = get_angle(ant_iniR, ant_endR)
        # Converting values into 0-180
        (angleL, prev_angL) = correctAngles(angleL, prev_angL)
        (angleR, prev_angR) = correctAngles(angleR, prev_angR)

        # """ Output """
        # File Output
        line = "\t" + str(count) + "\t\t" + str(ant_endL) + "\t\t" + str(ant_endR) + "\t\t" + str(angleL) + "\t\t" \
               + str(angleR) + "\t\t" + str(led_state) +"\n"
        fo.write(line)
        # Graphic Debug Red and Green circle colours of the point selected
        if DEBUG:

            cv2.circle(frame, ant_endL, 1, (0, 0, 255))
            cv2.circle(frame, ant_endR, 1, (0, 255, 0))
            cv2.imshow('Debug', frame)


            cv2.circle(blank_image, ant_endL, 30, (0, 0, 255))
            cv2.circle(blank_image, ant_endR, 30, (0, 255, 0))
            cv2.imshow('BE', fgmask500)
            cv2.imshow('LD', blank_image)
            cv2.waitKey()
            # Escape Button [Esc], for frame by frame mode replace with cv2.waitKey()
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        # """ Update Loop"""
        count += 1
	ret, frame = cap.read()    
    tt = time.clock()
    # Print time
    if DEBUG:
        print tt-t
    # Close open file
    fo.close()
    cap.release()
    if DEBUG:
        cv2.destroyAllWindows()
    print "Closing file " + input_name
    return


def extractPoints(blank_image, end_pl, end_pr):
    """     Look for point candidates in each of the direction x,y
        inputs: blank_image: image with Harris points
                end_pX:      distal point dummy
        return: detect_points in each of the direction x,y,xr,yr"""
    x_axis = list()
    y_axis = list()
    xr_axis = list()
    yr_axis = list()
    detect_points = list()
    # Select first from each dimension
    control_y = False
    control_x = False
    length_image = blank_image.shape

    leftList = list()
    rightList = list()
    topList = list()
    bottomList = list()

    # Extract limit points of the image
    img_grey = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)   # Convert to black/white
    ret,img_thresh = cv2.threshold(img_grey,127,255,0)		# Threshold
    unp,contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # FindContours

    for cnt in contours:		# Merge all contours points
        leftList.append(tuple(cnt[cnt[:,:,0].argmin()][0]))
        rightList.append(tuple(cnt[cnt[:,:,0].argmax()][0]))
        topList.append(tuple(cnt[cnt[:,:,1].argmin()][0]))
        bottomList.append(tuple(cnt[cnt[:,:,1].argmax()][0]))
    # Sort and choose first in each of the directions
    x = sorted(leftList, key=lambda x: x[0])[0]	
    xR = sorted(rightList, key=lambda x: x[0], reverse = True)[0]
    y = sorted(topList, key=lambda x: x[1])[0]	
    yR = sorted(bottomList, key=lambda x: x[1], reverse = True)[0]	


    if DEBUG:
        cv2.circle(blank_image, x, 30, (0, 0, 255))
        cv2.circle(blank_image, xR, 30, (0, 255, 0))
        cv2.circle(blank_image, y, 30, (255, 0, 255))
        cv2.circle(blank_image, yR, 30, (255,255,0))
    # Filtering by distances
    detect_points.append(x)
    if distance(xR, x) > LIMITSEP:
        detect_points.append(xR)
    if distance(y, x) > LIMITSEP and distance(y, xR) > LIMITSEP:
        detect_points.append(y)
    if distance(yR, x) > LIMITSEP and distance(yR, xR) > LIMITSEP and distance(
            yR, y) > LIMITSEP:
        detect_points.append(yR)

    return detect_points


def distance(p0, p1):
    """ Calculate Euclidean distance between 2 points"""
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def detection(points, mem_points):
    """ Detect minimun values of each tuple dimension"""
    minL  = 99999
    minR  = 99999
    left  = 99999
    right = 99999
    for p in points:
        disL = distance(p, mem_points.get("Left"))
        disR = distance(p, mem_points.get("Right"))
        if disL < minL:
            left = p
            minL = disL
        if disR < minR:
            right = p
            minR = disR
    # Preventing true negatives error according external value (maximun gap between samples)
    if distance(left, mem_points.get("Left")) > MaxVar:
        left = mem_points.get("Left")
    if distance(right, mem_points.get("Right")) > MaxVar:
        right = mem_points.get("Right")

    return (left, right)


def get_angle(p0, p1=np.array([0, 0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = (p0[0], p1[1])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return round(np.degrees(angle), 2)


def correctAngles(angle, prev_ang):
    ''' Convert Angles from 0-90 to 0-180 based on axis
    Inputs:
        angle: obtained angle at this state
        prev_ang: previous angle obtained in previous state
    '''
    if angle <= 0 and prev_ang >= 80 and prev_ang <= 90:
        angle_result = 90 + (90 + angle)
        prev_ang_result = 90
    elif angle >= 0 and prev_ang <= -80 and prev_ang >= -90:
        angle_result = -90 - (90 - angle)
        prev_ang_result = -90
    else:
        angle_result = angle
        prev_ang_result = angle
    return angle_result, prev_ang_result


if __name__ == "__main__":
    main()

