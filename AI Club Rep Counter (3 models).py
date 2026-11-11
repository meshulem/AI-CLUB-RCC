#=====================Instructions=====================

# see requirements.txt for how to get started

# Stand in front of your default webcam with your entire arm visible

# Your body should be perpendicular to the screen

# Do bicep curls and the reps should count up

# Press any key to finish


#=================Import requirements=================

import sys
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_hub as hub
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

#==================Key Choices==================

# Choose arm ("right" or "left")
rightOrLeft = "right"

# Choose model for detecting pose (1=MediaPipe, 2=OpenPose, 3=MoveNet)
# 2 needs pose_iter_440000.caffemodel. Download from https://huggingface.co/camenduru/openpose/blob/main/models/pose/coco/pose_iter_440000.caffemodel and put into folder
poseModel = 1

# Choose machine learning model for class prediction (1=Random Forest, 2=Support Vector Machine(SVM), 3=K-Nearest Neighbors (KNN), 4=Neural Network)
# Prediction models only work (even a little well) with poseModel=1
predictionModel = 3

# Choose Minimum angle to be counted as a rep
minimumAngle = 60

#==================Other Choices==================

#options for text of body parts
textColor = 255
textOffset = 15

#height of other text
textHeight = 48

minimumTimeBetweenReps = 0.5
minimumProminenceOfReps = 15 #see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences for how prominnce is calculated

timeToCountSecPerRep = 5

#size of averaging for counting reps
movingAverageSize = 3

predictionWhenUnsure = "..."

percentageOfFrameWidthToPlacePlotLine = 0.7

#==================MediaPipe Setup==================

if poseModel == 1:
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

#==================OpenPose Setup==================

net = None
if poseModel == 2:

    net = cv.dnn.readNet("pose_iter_440000.caffemodel", "pose_deploy.prototxt")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

    thresh = 0.1

    modelResolution = 100 #max is 368, the higher the slower the model

    BODY_PARTS_OPENPOSE = {
        "Nose": 0,
        "Neck": 1,
        "Right Shoulder": 2,
        "Right Elbow": 3,
        "Right Wrist": 4,
        "Left Shoulder": 5,
        "Left Elbow": 6,
        "Left Wrist": 7,
        "Right Hip": 8,
        "Right Knee": 9,
        "Right Ankle": 10,
        "Left Hip": 11,
        "Left Knee": 12,
        "Left Ankle": 13,
        "Right Eye": 14,
        "Left Eye": 15,
        "Right Ear": 16,
        "Left Ear": 17,
        "Background": 18,
    }

    # Will draw lines between these points
    POSE_PAIRS_OPENPOSE = [
        ["Neck", "Right Shoulder"],
        ["Neck", "Left Shoulder"],
        ["Right Shoulder", "Right Elbow"],
        ["Right Elbow", "Right Wrist"],
        ["Left Shoulder", "Left Elbow"],
        ["Left Elbow", "Left Wrist"],
        ["Neck", "Right Hip"],
        ["Right Hip", "Right Knee"],
        ["Right Knee", "Right Ankle"],
        ["Neck", "Left Hip"],
        ["Left Hip", "Left Knee"],
        ["Left Knee", "Left Ankle"],
        ["Neck", "Nose"],
        ["Nose", "Right Eye"],
        ["Right Eye", "Right Ear"],
        ["Nose", "Left Eye"],
        ["Left Eye", "Left Ear"],
    ]

#==================MoveNet Setup==================

if poseModel == 3:
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

    inputSize = 256 #256 for thunder (slower, better), 192 for lightning (faster, worse)

    BODY_PARTS_MOVENET = [
        "nose", 
        "Left Eye", 
        "right Eye", 
        "Left Ear", 
        "Right Ear",
        "Left Shoulder", 
        "Right Shoulder", 
        "Left Elbow", 
        "Right Elbow",
        "Left Wrist",
        "Right Wrist", 
        "Left Hip", 
        "Right Hip",
        "Left Knee", 
        "Right Knee", 
        "Left Ankle", 
        "Right Ankle"
    ]

    POSE_PAIRS_MOVENET = [
        ["Right Shoulder", "Right Elbow"],
        ["Right Elbow", "Right Wrist"],
        ["Left Shoulder", "Left Elbow"],
        ["Left Elbow", "Left Wrist"],
        ["Right Hip", "Right Knee"],
        ["Right Knee", "Right Ankle"],
        ["Left Hip", "Left Knee"],
        ["Left Knee", "Left Ankle"],
        ["Nose", "Right Eye"],
        ["Right Eye", "Right Ear"],
        ["Nose", "Left Eye"],
        ["Left Eye", "Left Ear"],
    ]

#===============Prediction Model Setup==================

match predictionModel:
    case 1:
        randomForest = joblib.load("Saved_Model_Prototype_RF.joblib")
    case 2:
        SVM = joblib.load("Saved_Model_Prototype_SVM.joblib")
    case 3:
        KNN = joblib.load("Saved_Model_Prototype_KNN.joblib")
    case 4: 
        NN = joblib.load("Saved_Model_Prototype_NN.joblib")

#================================================

#switches arm because image is flipped
if rightOrLeft == "right":
    rightOrLeft = "left"
elif rightOrLeft == "left":
    rightOrLeft = "right"

class PoseDetector():

    def __init__(self, staticImageMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False, smoothSegmentation=True, minDetectionConfidence=0.5, minTrackingConfidence=0.5):

        #the following is only used for MediaPipe

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
    
        match poseModel:
            case 1:
                img, armAngleRight, armAngleLeft = self.getAnglesMediaPipe(img, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)
            case 2:
                img, armAngleRight, armAngleLeft = self.getAnglesOpenPose(img, frame_width, frame_height, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)
            case 3:
                img, armAngleRight, armAngleLeft = self.getAnglesMoveNet(img, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)

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
    
    def getAnglesOpenPose(self, img, frame_width, frame_height, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist):

        blob = cv.dnn.blobFromImage(
            img, 1.0 / 255, (modelResolution, modelResolution), (0, 0, 0), swapRB=False, crop=False
        )
        net.setInput(blob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for i in range(len(BODY_PARTS_OPENPOSE)):
            heatMap = output[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H

            if conf > thresh:
                cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1)
                points.append((int(x), int(y)))

                cv.putText(img, list(BODY_PARTS_OPENPOSE.keys())[i], (int(x)+textOffset, int(y)+textOffset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (textColor, textColor, textColor))

                if list(BODY_PARTS_OPENPOSE.keys())[i] == "Right Shoulder":
                    r_shoulder[0] = x
                    r_shoulder[1] = y
                elif list(BODY_PARTS_OPENPOSE.keys())[i] == "Left Shoulder":
                    l_shoulder[0] = x
                    l_shoulder[1] = y
                elif list(BODY_PARTS_OPENPOSE.keys())[i] == "Right Elbow":
                    r_elbow[0] = x
                    r_elbow[1] = y
                elif list(BODY_PARTS_OPENPOSE.keys())[i] == "Left Elbow":
                    l_elbow[0] = x
                    l_elbow[1] = y
                elif list(BODY_PARTS_OPENPOSE.keys())[i] == "Right Wrist":
                    r_wrist[0] = x
                    r_wrist[1] = y
                elif list(BODY_PARTS_OPENPOSE.keys())[i] == "Left Wrist":
                    l_wrist[0] = x
                    l_wrist[1] = y
            else:
                points.append(None)

        for pair in POSE_PAIRS_OPENPOSE:
            partFrom = pair[0]
            partTo = pair[1]

            idFrom = BODY_PARTS_OPENPOSE[partFrom]
            idTo = BODY_PARTS_OPENPOSE[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(img, points[idFrom], points[idTo], (0, 255, 0), 2)

                cv.circle(img, points[idFrom], 5, (0, 0, 255), -1)
                cv.circle(img, points[idTo], 5, (0, 0, 255), -1)

        armAngleRight, armAngleLeft = self.calculateAnglesWithVectors(armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist)

        return img, armAngleRight, armAngleLeft
    
    def getAnglesMoveNet(self, img, armAngleRight, armAngleLeft, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist):

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # MoveNet expects RGB, OpenCV gives BGR
        input_image = tf.image.resize_with_pad(tf.expand_dims(rgb, axis=0), inputSize, inputSize)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run pose estimation
        outputs = movenet.signatures['serving_default'](input=input_image)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]

        # Process the detected keypoints
        h, w, _ = img.shape
        adjustedKeypoints = keypoints * [h, w, 1]

        processedKeypoints = {}
        for idx, keypoint in enumerate(adjustedKeypoints):
            if keypoint[2] > 0.5:  # Confidence threshold
                processedKeypoints[BODY_PARTS_MOVENET[idx]] = (keypoint[1], keypoint[0])

        # Draw keypoints on the frame
        for key, coord in processedKeypoints.items():

            x = int(coord[0])
            y = int(coord[1])

            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv.putText(img, key, (x+textOffset, y+textOffset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (textColor, textColor, textColor))

            if key == "Right Shoulder":
                r_shoulder[0] = x
                r_shoulder[1] = y
            elif key == "Left Shoulder":
                l_shoulder[0] = x
                l_shoulder[1] = y
            elif key == "Right Elbow":
                r_elbow[0] = x
                r_elbow[1] = y
            elif key == "Left Elbow":
                l_elbow[0] = x
                l_elbow[1] = y
            elif key == "Right Wrist":
                r_wrist[0] = x
                r_wrist[1] = y
            elif key == "Left Wrist":
                l_wrist[0] = x
                l_wrist[1] = y

        for pair in POSE_PAIRS_MOVENET:
            partFrom = pair[0]
            partTo = pair[1]

            if partFrom in processedKeypoints and partTo in processedKeypoints:
                pt1 = (int(processedKeypoints[partFrom][0]), int(processedKeypoints[partFrom][1]))
                pt2 = (int(processedKeypoints[partTo][0]), int(processedKeypoints[partTo][1]))
                
                cv.line(img, pt1, pt2, (0, 255, 0), 2)

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

class ClassPredictor():
    def getPrediction(self, angleSample):

        angleSample = np.multiply(angleSample, 1/180)
        

        #Gets a numercal prediction (-1 if unsure)
        match predictionModel:
            case 1:
                numericalPrediction = self.getPredictionRandomForest(angleSample)
            case 2:
                numericalPrediction = self.getPredictionSVM(angleSample)
            case 3:
                numericalPrediction = self.getPredictionKNN(angleSample)
            case 4:
                numericalPrediction = self.getPredictionNeuralNetwork(angleSample)

        # Writes a prediction based on numercal prediction
        match numericalPrediction:
            case 1:
                writtenPrediction = "Good"
            case 0:
                writtenPrediction = "Bad"
            case -1:
                writtenPrediction = predictionWhenUnsure

        return writtenPrediction
    
    def getPredictionRandomForest(self, angleSample):
        quality = randomForest.predict([angleSample])[0]
        qualityPercentages = randomForest.predict_proba([angleSample])[0]

        #Gets quality if model is confident
        if max(qualityPercentages) > 0.6:
            prediction = quality
        else:
            prediction = -1

        return prediction
    
    def getPredictionSVM(self, angleSample):
        quality = SVM.predict([angleSample])[0]
        qualityPercentages = SVM.predict_proba([angleSample])[0]

        #Gets quality if model is confident
        if max(qualityPercentages) > 0.6:
            prediction = quality
        else:
            prediction = -1

        return prediction
    
    def getPredictionKNN(self, angleSample):
        quality = KNN.predict([angleSample])[0]
        qualityPercentages = KNN.predict_proba([angleSample])[0]
        
        #Gets quality if model is confident
        if max(qualityPercentages) > 0.6:
            prediction = quality
        else:
            prediction = -1

        return prediction
    
    def getPredictionNeuralNetwork(self, angleSample):
        
        angleSample = np.array(angleSample).reshape(1, -1)

        qualityPercentages = NN.predict(angleSample)
        quality = tf.argmax(qualityPercentages, axis=-1).numpy()

        #Gets quality if model is confident
        if max(qualityPercentages[0]) > 0.6:
            prediction = quality
        else:
            prediction = -1

        return prediction
    

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

    predictor = ClassPredictor()

    prediction = predictionWhenUnsure

    fpsfinalTime = 0
    fpsinitTime = 0

    reps = 0
    secPerRep = 0
    repsArray = [0,0]

    target_fps = 100
    wait_key_delay = int(1000 / target_fps)

    armAngleLeftArray = [0]
    armAngleRightArray = [0]
    timeArray = [0]

    initTime = time.time()

    frames = 0

    XOrigin = int(percentageOfFrameWidthToPlacePlotLine*frame_width)

    while True:

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
        fpsinitTime = fpsfinalTime

        timeArray.append(fpsfinalTime - initTime)

        if rightOrLeft == "right":
            angle = armAngleRightArray
        elif rightOrLeft == "left":
            angle = armAngleLeftArray

        plot = cv.line(
            plot, 
            (XOrigin, int(angle[-2] * angleToYRatio)), 
            (int(XOrigin  + (timeArray[-1] - timeArray[-2]) * timeToXRatio), int(angle[-1] * angleToYRatio)), 
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

        smoothedAngles = uniform_filter1d(angle, size = movingAverageSize)

        flippedSmoothedAngles = np.flip(180 - smoothedAngles) #smoothedAngles is flipped twice: vertically so that peaks are contractions, horizontally so that circle of rep can be drawn easier

        peaks, _ = find_peaks(flippedSmoothedAngles, height=180-minimumAngle, distance=max(1, minimumTimeBetweenReps*fps), prominence=minimumProminenceOfReps) 

        reps = len(peaks)

        repsArray.append(reps)

        #draws circle on new rep (if there is one)
        if repsArray[-1] > repsArray[-2]:

            x = int(XOrigin - (timeArray[-1] - timeArray[-peaks[0]]) * timeToXRatio)
            y = int(angle[-peaks[0]] * angleToYRatio)

            cv.circle(plot, (x, y), 15, (255, 0, 0), -1)
            cv.circle(plot, (x, y), 15, (255, 255, 255), 8)

        #calculates seconds per rep
        if frames > 110:
            try:
                secPerRep = max(0,(timeArray[-1] - timeArray[-100])/(repsArray[-1] - repsArray[-100]))
            except:
                secPerRep = 0
            
        # Guesses quality based on last 100 angles
        if frames > 110:
            prediction = predictor.getPrediction(angle[-100:])

        #Draws rectangles and text
        cv.rectangle(canvas, (1, 1), (int(5.3*textHeight), int(1.3*textHeight)), (255, 255, 255), -1)
        cv.rectangle(canvas, (1, 1), (int(5.3*textHeight), int(1.3*textHeight)), (255, 0, 0), 5)
        cv.rectangle(canvas, (1, frame_height), (int(11.1*textHeight), int(frame_height+4.3*textHeight)), (255, 255, 255), -1)
        cv.rectangle(canvas, (1, frame_height), (int(11.1*textHeight), int(frame_height+4.3*textHeight)), (255, 0, 0), 5)

        cv.putText(canvas, "FPS: " + str(int(fps)), (2, textHeight), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv.putText(canvas, "Reps: " + str(int(reps)), (4, frame_height+textHeight+4), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv.putText(canvas, "Seconds Per Rep: " + str(round(secPerRep, 1)), (4, frame_height+2*textHeight+4), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(canvas, "Prediction: " + str(prediction), (4, int(frame_height+3.5*textHeight+4)), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv.imshow("Image", canvas)

        key = cv.waitKey(wait_key_delay)
        if not key == -1 or cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()

    print("=========================================")

    if rightOrLeft == "right":
        with open("angle.txt", "w", encoding="utf-8") as f:
            f.write(" ".join([str(s) for s in armAngleRightArray]))

    elif rightOrLeft == "left":
        with open("angle.txt", "w", encoding="utf-8") as f:
            f.write(" ".join([str(s) for s in armAngleLeftArray]))

    print("=========================================")

    with open("timeArray.txt", "w", encoding="utf-8") as f:
        f.write(" ".join([str(s) for s in timeArray]))

    print("=========================================")

if __name__ == '__main__':
    main()
