import cv2
import numpy as np
import math

# Change the below path to the directory containing the frames extracted after executing the frame_extraction.py file
DirectoryPath = "/Users/mehulmathur/Desktop/Main Folder 0/Python Files/computer_vision_workshop/Final Project Folder/frames/frames1"

count = 0
image = cv2.imread(f"{DirectoryPath}/frame_{count}.jpg")
minPts = 12
pt1 = (0,0)
pt2 = (0,0)

height, width, _ = np.shape(image)

fgbg = cv2.createBackgroundSubtractorMOG2()
ker = np.ones((3, 3), np.uint8)
rhoMax = int(math.sqrt(pow(width, 2) + pow(height, 2)))
thetaMax = 180      # in degrees
maxi = 0

def houghLines(points, height, width):
    arr = np.zeros((2*rhoMax, thetaMax), np.uint8)          # rho, theta (in deg)

    # as line is rho = x cos(theta) + y sin(theta)
    maxn = 1
    for coordinate in points:
        j, i = coordinate
        for theta in range(thetaMax):
            rho = int(j*math.cos(theta*math.pi/180) + i*math.sin(theta*math.pi/180))
            arr[rhoMax + rho, theta] += 1
            if (arr[rhoMax + rho, theta] > maxn):
                maxn = arr[rhoMax+rho, theta]
                maxi = i
                reqRho = rho
                reqTheta = theta

    pts = [(0,0), (0,0)]
    if maxn >2:
        if (reqTheta == 0): reqTheta = 1e-9
        slope = -1/math.tan(math.radians(reqTheta))
        c = int(reqRho/math.sin(math.radians(reqTheta)))
        pts = []
        height = maxi
        ptBottom = (int((height-1 - c)/slope), height-1)
        ptLeft = (0, int(c))
        ptTop = (int(-c/slope) , 0)
        ptRight = (width-1, int(slope*(width-1) + c))
        if (ptTop[0] <= width and ptTop[0] >= 0): pts.append(ptTop)
        if (ptBottom[0] <= width and ptBottom[0] >= 0): pts.append(ptBottom)
        if (ptLeft[1] <= height and ptLeft[1] >= 0): pts.append(ptLeft)
        if (ptRight[1] <= height and ptRight[1] >= 0): pts.append(ptRight)

    return (pts[0], pts[1], maxn)

def EdgeDetector(image):
    blurGrayImg = cv2.GaussianBlur(image, (3, 3), 0)

    gradX = cv2.Sobel(blurGrayImg, cv2.CV_16S, 1, 0, ksize=3)
    gradY = cv2.Sobel(blurGrayImg, cv2.CV_16S, 0, 1, ksize=3)

    gradXabs = cv2.convertScaleAbs(gradX)
    gradYabs = cv2.convertScaleAbs(gradY)

    imageNet = cv2.addWeighted(gradXabs, 0.5, gradYabs, 0.5, 0)

    return imageNet

while(True):
    points = []
    img = cv2.imread(f"{DirectoryPath}/frame_{count}.jpg")
    fgmask = fgbg.apply(img)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, ker, iterations = 4)
    opening_ = cv2.blur(opening, (4, 4))
    opening_ = cv2.threshold(opening_, 200, 255, cv2.THRESH_BINARY)[1]
    imagefG = EdgeDetector(opening_)

    for i in range(0,height, 5):
        xmin = 1e6
        xmax = 0
        for j in range(width-1):
            if (imagefG[i][j]  != imagefG[i][j+1]):
                xmin = min(xmin, j)
                xmax = max(xmax, j)
        if (xmax != 0):
            points.append((int((xmin+xmax)/2), i))

    p1, p2, maxn = houghLines(points, height, width)
    if maxn > minPts: 
        pt1, pt2 = (p1, p2)
        print(maxn)
    cv2.line(img, pt1, pt2, 255, 6)

    # NOTE:

    cv2.imwrite("frame_"+str(count)+".jpg", img)    # To store the frames in the current working directory, uncomment this     

    # cv2.imshow("final", img)                # To show the realtime output, don't comment out these
    # if cv2.waitKey(1) & 0xFF==ord('q'):     # but comment out the line with cv2.imwrite
    #    break                                #

    count+=1