#==================Instructions==================

# Set a first trial and a classifier. Time will countdown from 5, after which angles will be recorded for a brief while. 
# Then, the time will coundtown from 5 again and the next trial will begin.
# Press any key any time to finish.

#=================Import requirements=================

import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import time
import pandas as pd
import csv
import string

#==================Key Choices==================

# Choose arm ("right" or "left")
rightOrLeft = "right"

firstTrial = 41

classifier = 1

fileName = "Training_Protype.csv"

#==================Other Choices==================

#options for text of body parts
textColor = 255
textOffset = 15

#height of other text
textHeight = 48

movingAverageSize = 3

#==================MediaPipe Setup==================
#These are the ids for the body parts taken from mediapipe. 0 is nose, 1 is Left Eye (inner), etc.
BODY_PARTS_MEDIAPIPE = [
    "Nose",
    "Left Eye (inner)",
    "Left Eye",
    "Left Eye (outer)",
    "Right Eye (inner)",
    "Right Eye",
    "Right Eye (outer)",
    "Left Ear",
    "Right Ear",
    "Mouth (left)",
    "Mouth (right)",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Pinky",
    "Right Pinky",
    "Left Index",
    "Right Index",
    "Left Thumb",
    "Right Thumb",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
    "Left Heel",
    "Right Heel",
    "Left Foot Index",
    "Right Foot Index",
]

#switches arm because image is flipped
if rightOrLeft == "right":
    rightOrLeft = "left"
elif rightOrLeft == "left":
    rightOrLeft = "right"

class PoseDetector():

    def __init__(self, staticImageMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False, smoothSegmentation=True, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.staticImageMode = staticImageMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.staticImageMode, self.modelComplexity, self.smoothLandmarks, self.enableSegmentation, self.smoothSegmentation, self.minDetectionConfidence, self.minTrackingConfidence)
    
    def getAngles(self, img, frame_width, frame_height, armAngleRightArray, armAngleLeftArray):

        #sets the angles to the previous one, so that if the angle is not detected, line continues horizontally
        armAngleRight = armAngleRightArray[-1]
        armAngleLeft = armAngleLeftArray[-1]

        #sets up empty arrays for parts to be filled in later
        r_shoulder = [0,0]
        l_shoulder = [0,0]
        r_elbow = [0,0]
        l_elbow = [0,0]
        r_wrist = [0,0]
        l_wrist = [0,0]
    
        img, armAngleRight, armAngleLeft = self.getAnglesMediaPipe(img, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)

        return img, armAngleRight, armAngleLeft
    
    def getAnglesMediaPipe(self, img, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if BODY_PARTS_MEDIAPIPE[id] == "Right Shoulder":
                    r_shoulder[0] = cx
                    r_shoulder[1] = cy
                elif BODY_PARTS_MEDIAPIPE[id] == "Left Shoulder":
                    l_shoulder[0] = cx
                    l_shoulder[1] = cy
                elif BODY_PARTS_MEDIAPIPE[id] == "Right Elbow":
                    r_elbow[0] = cx
                    r_elbow[1] = cy
                elif BODY_PARTS_MEDIAPIPE[id] == "Left Elbow":
                    l_elbow[0] = cx
                    l_elbow[1] = cy
                elif BODY_PARTS_MEDIAPIPE[id] == "Right Wrist":
                    r_wrist[0] = cx
                    r_wrist[1] = cy
                elif BODY_PARTS_MEDIAPIPE[id] == "Left Wrist":
                    l_wrist[0] = cx
                    l_wrist[1] = cy

                cv.putText(img, BODY_PARTS_MEDIAPIPE[id], (cx+15, cy+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (textColor, textColor, textColor))
                cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)

        armAngleRight, armAngleLeft = self.calculateAnglesWithVectors(armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)

        return img, armAngleRight, armAngleLeft

    def calculateAnglesWithVectors(self, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist):
        if not (r_shoulder == [0,0] or r_elbow == [0,0] or r_wrist == [0,0]):

            r_shoulder = np.array(r_shoulder)
            r_wrist = np.array(r_wrist)

            r_shoulder = np.subtract(r_shoulder, r_elbow)
            r_wrist = np.subtract(r_wrist, r_elbow)

            dotProduct = np.dot(r_wrist, r_shoulder)

            magnitudesMultiplication = np.linalg.norm(r_wrist)*np.linalg.norm(r_shoulder)

            if magnitudesMultiplication >0:
                cosineOfAngle = dotProduct/magnitudesMultiplication
                armAngleRightRad = np.arccos(cosineOfAngle)
                armAngleRight = np.degrees(armAngleRightRad)

        if not (l_shoulder == [0,0] or l_elbow == [0,0] or l_wrist == [0,0]):

            l_shoulder = np.array(l_shoulder)
            l_wrist = np.array(l_wrist)

            l_shoulder = np.subtract(l_shoulder, l_elbow)
            l_wrist = np.subtract(l_wrist, l_elbow)

            dotProduct = np.dot(l_wrist, l_shoulder)
            magnitudesMultiplication = np.linalg.norm(l_wrist)*np.linalg.norm(l_shoulder)

            if magnitudesMultiplication >0:
                cosineOfAngle = dotProduct/magnitudesMultiplication
                armAngleLeftRad = np.arccos(cosineOfAngle)
                armAngleLeft = np.degrees(armAngleLeftRad)

        return armAngleRight, armAngleLeft
    
    def writeTrial(self, trial, classifier, angleSample):
        array = []

        array = np.loadtxt(fileName, delimiter=",", dtype=str)

        #adds row if needed
        try:
            array[trial][0] = trial
        except:
            array = np.r_[array,[[trial,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]

        array[trial][1] = classifier

        array[trial][2:102] = angleSample[-100:]

        with open(fileName, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(array)

def main():
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    cap = cv.VideoCapture(s)

    fps = 15

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    plot = np.zeros((int(frame_height*0.5), int(frame_width), 3), dtype = np.uint8)
    timeToXRatio = frame_width/20
    angleToYRatio = int(frame_height*0.5)/180

    canvas = np.zeros((int(frame_height*1.5), int(frame_width), 3), dtype = np.uint8)

    detector = PoseDetector()

    fpsfinalTime = 0
    fpsinitTime = time.time()

    target_fps = 100
    wait_key_delay = int(1000 / target_fps)

    armAngleLeftArray = [0]
    armAngleRightArray = [0]
    timeArray = [0]

    initTime = time.time()

    trial = firstTrial

    frames = 0

    countdown = 10

    while True:

        if countdown == 0:
            frames+=1

        success, img = cap.read()
        if not success:
            break
        
        img = cv.flip(img, 1)

        img, armAngleRight, armAngleLeft = detector.getAngles(img, frame_width, frame_height, armAngleRightArray, armAngleLeftArray)

        armAngleRightArray.append(int(armAngleRight))
        armAngleLeftArray.append(int(armAngleLeft))

        fpsfinalTime = time.time()
        fps = 1 / (fpsfinalTime - fpsinitTime)
        if countdown != 0:
            countdown = countdown - (fpsfinalTime - fpsinitTime)
        fpsinitTime = fpsfinalTime

        print(fpsfinalTime - fpsinitTime)

        timeArray.append(fpsfinalTime - initTime)

        if rightOrLeft == "right":
            angle = armAngleRightArray
        elif rightOrLeft == "left":
            angle = armAngleLeftArray

        plot = cv.line(
            plot, 
            (int(frame_width/2), int(angle[-2] * angleToYRatio)), 
            (int(frame_width/2  + (timeArray[-1] - timeArray[-2]) * timeToXRatio), int(angle[-1] * angleToYRatio)), 
            (0, 255, 0), 
            2
        )

        try:
            pixelstoShift = int((timeArray[-1] - timeArray[-2]) * timeToXRatio)
        except:
            pixelstoShift = 0

        plot[:, :-pixelstoShift] = plot[:, pixelstoShift:]
        plot[:, -pixelstoShift:] = 0

        canvas[:frame_height, :] = img
        canvas[frame_height:, :] = plot[:, (-frame_width):]

        #Draws rectangle and text for frames
        cv.rectangle(canvas, (1, 1), (int(9*textHeight), int(2.5*textHeight)), (255, 255, 255), -1)
        cv.rectangle(canvas, (1, 1), (int(9*textHeight), int(2.5*textHeight)), (255, 0, 0), 5)

        if countdown > 0:
            cv.putText(canvas, "Starting in: " + str(int(countdown)), (2, textHeight), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            countdown = 0
            cv.putText(canvas, "Frames: " + str(int(frames)), (2, textHeight), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv.putText(canvas, "Trial: " + str(int(trial)), (2, int(2.2*textHeight)), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        

        cv.imshow("Image", canvas)

        if frames>101:
            frames = 0
            countdown = 5

            detector.writeTrial(trial, classifier, angle)

            trial += 1

        key = cv.waitKey(wait_key_delay)
        if not key == -1 or cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()