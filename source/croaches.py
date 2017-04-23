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


"""         Global Parameters        """
# Visual Mode
DEBUG = 0
# Video Characteristics
# Filtering by distance
Filtering = False   # Filter part of the frame
N = 3               # It will draw lines whose points exceed from the 1/N of the image
# Static Object
Static = True       # If Static is False it apply minimun length to the lines in order to filter legs movement
#   Non Static
Minline_length = 100
Maxline_gap = 0
Threshold = 50      # Threshold of connected trajectories in order to detect a line
#   Params Static Case
Minline_length_static = 50
Maxline_gap_static = 0
Threshold_static = 50
# Point Extraction Variables
LIMITSEP = 50       # Minimal separation between points Isolation point function
MaxVar = 200        # Maximal distance allowed between frames
# LED Detection
SizeWin = 200       # Length in pixels of window around the led in order to Detect
LedStart = 1        # State of Led at the beginning
LedThreshold = 15000
# File Directories
InputDirectory = False
Current_folder = os.path.dirname(os.path.abspath(__file__))
Input_directory = os.path.join(Current_folder, 'src','170411')
Output_directory = os.path.join(Current_folder, 'out','170411')
File = "170603_right_peppermint_double_pulse_1_20170306_171931_20170306_172132sh.avi"#"170302_front_apple_long_pulse_1_20170302_181221_20170302_181338.avi"

"""                                   """

#              Image => (0,0)xxxxxxxxxxxxxxx(i,0)
#                       (0,j/N)xxxxxxxxxxx(i,j/N)
#                       (0,j)xxxxxxxxxxxxxxx(i,j)


def main():
    # Graphical mode activation
    if DEBUG:
        cv2.namedWindow("BE", cv2.WINDOW_NORMAL)
        cv2.namedWindow("LD", cv2.WINDOW_NORMAL)
    # Batch Tracking
    if InputDirectory:
        allfiles = [f for f in os.listdir(Input_directory) if os.path.isfile(os.path.join(Input_directory, f))]
        init_points = list()
        distal_points = list()
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
            plt.close()
            init_points.append((ant_iniL, ant_iniR))
            distal_points.append((end_pL, end_pR))

        for a in allfiles:
            in_name = os.path.join(Input_directory, a)
            out_name = os.path.join(Output_directory, a.split('.')[0] + '.txt')
            trackVideo(in_name, out_name, init_points.pop(0), distal_points.pop(0))
    else:
        in_name = os.path.join(Input_directory, File)
        out_name = os.path.join(Output_directory, File.split('.')[0] + '.txt')
        # """ Selecting points from the image """
        cap = cv2.VideoCapture(in_name)
        ret, frame = cap.read()  # Initialize Stream
        #   Fetching Filename Control
        if not ret:
            print "File " + in_name + " Not found"
            return

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

        ax.set(title='Click in LED position')
        plt.imshow(frame)
        (point) = plt.ginput(1)
        point = (int(point[0][0]), int(point[0][1]))
        plt.close()
        init_p = (ant_iniL, ant_iniR)
        distal_p = (end_pL, end_pR)

        trackVideo(in_name, out_name, init_p, distal_p, point)
    return




def trackVideo(input_name, output_name, init_points, distal_points, point):
    """
        Main function of tracking
        Inputs a video file
        Outputs a sequence position of points correlated by lines
    """
    print "Reading file " + input_name
    # """ Initialize variables """
    ant_iniL = init_points[0]
    ant_iniR = init_points[1]
    end_pL   = distal_points[0]
    end_pR   = distal_points[1]
    count = 0                           # Count loop for file manipulation
    prev_angL = 0                      # Initialize memory previous angle Left
    prev_angR = 0                      # Initialize memory previous angle Right
    led_state = LedStart               # Inital state of the LED
    mem_points = {"Left": end_pL, "Right": end_pR}
    #   Dynamic Background Extraction Properties
    fgbg500 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
    cap = cv2.VideoCapture(input_name)    # Video Capture and Windowing
    ret, frame = cap.read()             # Initialize Stream

    #   Open Output points file
    fo = open(output_name, "wb")
    fo.write(" \tSample\t\tLeft\t\t\tRight\t\t\tAngleLeft\tAngleRight\tLedState\n")
    #   Use first image as Background
    fgmask500 = fgbg500.apply(frame)
    fgmask500 = cv2.dilate(fgmask500, kernel=np.ones((5, 5)), iterations=1)  # Clean image extraction
    crop_img = frame[max(0, point[1] - SizeWin / 2):min(point[1] + SizeWin / 2, frame.shape[0]),
               max(0, point[0] - SizeWin / 2):min(point[0] + SizeWin / 2,
                                                  frame.shape[1])]  # Crop from x, y, w, h -> 100, 200, 300, 400
    sumPix = sum(cv2.sumElems(crop_img))

    # """ Infinite loop until no more images in video buffer """
    while (ret):
        # Fetch new Frame
        ret, frame = cap.read()

        # """ Crop and compare LED Image """
        crop_img = frame[ max(0,point[1] - SizeWin/2):min(point[1] + SizeWin/2 , frame.shape[0] ), max(0, point[0] - SizeWin/2):min(point[0] + SizeWin/2, frame.shape[1] )]  # Crop from x, y, w, h -> 100, 200, 300, 400
        if sum(cv2.sumElems(crop_img)) - sumPix > LedThreshold:
            led_state += 1
        elif sum(cv2.sumElems(crop_img)) - sumPix < -LedThreshold:
            led_state -= 1
        sumPix = sum(cv2.sumElems(crop_img))
        # See Image cropped
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey()

        #  """ Move Detection """
        #   Background Extraction (See above for params)
        fgmask500 = fgbg500.apply(frame)
        fgmask500 = cv2.dilate(fgmask500, kernel=np.ones((5, 5)), iterations=1)  # Clean image extraction


        #  """ Line Detection """
        blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        #   Get Lines
        if Static:
            lines = cv2.HoughLinesP(fgmask500, 1, np.pi / 180, Threshold_static, lines=None,
                                    minLineLength=Minline_length_static, maxLineGap=Maxline_gap_static)
        else:
            lines = cv2.HoughLinesP(fgmask500, 1, np.pi / 180, Threshold, lines=None, minLineLength=Minline_length,
                                    maxLineGap=Maxline_gap)
        #   Draw Lines
        try:
            for draw in lines:
                for x1, y1, x2, y2 in draw:
                    if Filtering:
                        if y1 > frame.shape[0] / N and y2 > frame.shape[0] / N:
                            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    else:
                        cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        except:
            pass

        # """ Harris Corner Detection """
        gray_depth = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        gray_depth = np.float32(gray_depth)
        corn_ind = cv2.cornerHarris(gray_depth, 2, 3, 0.1)
        corn_ind = cv2.dilate(corn_ind, None)
        #   Threshold on image for an optimal value.
        blank_image[corn_ind > 0.01 * corn_ind.max()] = [0, 0, 255]

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
        # Graphic Debug
        if DEBUG:
            cv2.imshow('BE', fgmask500)
            cv2.imshow('LD', blank_image)
            # Escape Button [Esc]
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
                # Frame by frame cv2.waitKey()
        # """ Update counter """
        count += 1

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
        # X Axis
    for j in range(length_image[1]):
        for i in range(length_image[0]):
            if blank_image[i][j][0] != 0 or blank_image[i][j][1] != 0 or blank_image[i][j][2] != 0:
                x_axis.append((j, i))
                control_x = True
        if control_x:
            break
        # Y Axis
    for i in range(length_image[0]):
        for j in range(length_image[1]):
            if blank_image[i][j][0] != 0 or blank_image[i][j][1] != 0 or blank_image[i][j][2] != 0:
                y_axis.append((j, i))
                control_y = True
        if control_y:
            break
    control_y = False
    control_x = False
        # Reversed X Axis
    for j in reversed(range(length_image[1])):
        for i in range(length_image[0]):
            if blank_image[i][j][0] != 0 or blank_image[i][j][1] != 0 or blank_image[i][j][2] != 0:
                xr_axis.append((j, i))
                control_x = True
        if control_x:
            break
        # Reversed Y Axis
    for i in range(length_image[0]):
        for j in reversed(range(length_image[1])):
            if blank_image[i][j][0] != 0 or blank_image[i][j][1] != 0 or blank_image[i][j][2] != 0:
                yr_axis.append((j, i))
                control_y = True
        if control_y:
            break
    if not x_axis:
        x_axis.append(end_pl)
    if not y_axis:
        y_axis.append(end_pr)
    if not xr_axis:
        xr_axis.append(end_pr)
    if not yr_axis:
        yr_axis.append(end_pl)
    point1 = x_axis[0]  # if xAxis else distalPoint
    point2 = y_axis[0]  # if yAxis else distalPoint
    point3 = xr_axis[0]  # if xRAxis else distalPoint
    point4 = yr_axis[0]  # if yRAxis else distalPoint

    # Filtering by distances
    if x_axis:
        cv2.circle(blank_image, point1, 30, (0, 0, 255))
        detect_points.append(point1)
    if y_axis and distance(point1, point2) > LIMITSEP:
        cv2.circle(blank_image, point2, 30, (0, 255, 0))
        detect_points.append(point2)
    if xr_axis and distance(point1, point3) > LIMITSEP and distance(point3, point2) > LIMITSEP:
        cv2.circle(blank_image, point3, 30, (255, 0, 255))
        detect_points.append(point3)
    if yr_axis and distance(point1, point4) > LIMITSEP and distance(point4, point2) > LIMITSEP and distance(
            point4, point3) > LIMITSEP:
        cv2.circle(blank_image, point4, 30, (255, 255, 0))
        detect_points.append(point4)

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