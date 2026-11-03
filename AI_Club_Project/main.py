#Returns array every frame formatted as [[Body Part ID, x position, y position], [another Body Part ID, x position, y position], ...]

#Currently only uses mediapipe. Implement OpenPose and MoveNet.

import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import time

#Ids for body parts, so id of 1 is "Left Eye (inner)". Not used currently, but will become useful.
BODY_PARTS = [
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
    
    def findPose(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            lmList.append([id, cx, cy])

            if draw:
                cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)

        return lmList

    
def main():
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    cap = cv.VideoCapture(s)

    detector = PoseDetector()

    pTime = 0
    cTime = 0

    target_fps = 100
    wait_key_delay = int(1000 / target_fps)

    timeArray = [0]

    initTime = time.time()

    while True:
        success, img = cap.read()
        
        img = cv.flip(img, 1)

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:    
            print(lmList)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        timeArray.append(pTime - initTime)

        cv.putText(img, str(int(fps)), (1, 24), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv.imshow("Image", img)

        key = cv.waitKey(wait_key_delay)

        if key == ord("Q") or key == ord("q") or key == 27 or cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()