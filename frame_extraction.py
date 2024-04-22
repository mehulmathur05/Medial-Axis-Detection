import cv2 
import numpy as np

## This program extracts the frames on the vidPath and stores them in the current working directory
## Thus, change the terminal directory to the directory in which the frames need to be extracted and execute this file
def frame_capture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    flag = True
    while flag:
        flag, image = vidObj.read()
        cv2.imwrite("frame_%d.jpg" % count, image)
        count+=1

# change the path to the path of the video file
vidPath = "/Users/mehulmathur/Desktop/Main Folder 0/Python Files/computer_vision_workshop/Final Project Folder/3.mp4"
frame_capture(vidPath)