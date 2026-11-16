#=====================Instructions=====================

# see requirements.txt for how to get started

# Stand in front of your default webcam with your entire arm visible

# Your body should be perpendicular to the screen

# Do bicep curls and the reps should count up

# Press any key to finish


#=================Import requirements=================

import sys
import threading
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
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av
import pandas as pd
import altair as alt
import threading

#==================Key Choices==================

# Choose arm ("right" or "left")
rightOrLeft = "left"

# Choose model for detecting pose (1=MediaPipe, 2=OpenPose, 3=MoveNet)
poseModel = 1

# Choose machine learning model for class prediction (1=Random Forest, 2=Support Vector Machine(SVM), 3=K-Nearest Neighbors (KNN), 4=Neural Network)
# Prediction models only work (even a little well) with poseModel=1
predictionModel = 1

# Choose Minimum angle to be counted as a rep
minimumAngle = 60

secondsBetweenModelPredictions = 3

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

MinimumConfidenceOfPrediction = 0.5 #0.5-1

predictionWhenUnsure = "..."

percentageOfFrameWidthToPlacePlotLine = 0.7

#==================MediaPipe Setup==================

if poseModel == 1:
    #These are the ids for the body parts taken from mediapipe. 0 is nose, 1 is Right Eye (inner), etc.
    #Note that righta nd left are switched for all pose models becuase image is flipped
    BODY_PARTS_MEDIAPIPE = [
        "Nose",
        "Right Eye (inner)",
        "Right Eye",
        "Right Eye (outer)",
        "Left Eye (inner)",
        "Left Eye",
        "Left Eye (outer)",
        "Right Ear",
        "Left Ear",
        "Mouth (right)",
        "Mouth (left)",
        "Right Shoulder",
        "Left Shoulder",
        "Right Elbow",
        "Left Elbow",
        "Right Wrist",
        "Left Wrist",
        "Right Pinky",
        "Left Pinky",
        "Right Index",
        "Left Index",
        "Right Thumb",
        "Left Thumb",
        "Right Hip",
        "Left Hip",
        "Right Knee",
        "Left Knee",
        "Right Ankle",
        "Left Ankle",
        "Right Heel",
        "Left Heel",
        "Right Foot Index",
        "Left Foot Index",
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
        "Left Shoulder": 2,
        "Left Elbow": 3,
        "Left Wrist": 4,
        "Right Shoulder": 5,
        "Right Elbow": 6,
        "Right Wrist": 7,
        "Left Hip": 8,
        "Left Knee": 9,
        "Left Ankle": 10,
        "Right Hip": 11,
        "Right Knee": 12,
        "Right Ankle": 13,
        "Left Eye": 14,
        "Right Eye": 15,
        "Left Ear": 16,
        "Right Ear": 17,
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
    movenet = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")

    inputSize = 256 #256 for thunder (slower, better), 192 for lightning (faster, worse)

    BODY_PARTS_MOVENET = [
        "nose", 
        "Right Eye", 
        "Left Eye", 
        "Right Ear",
        "Left Ear", 
        "Right Shoulder", 
        "Left Shoulder", 
        "Right Elbow",
        "Left Elbow", 
        "Right Wrist", 
        "Left Wrist",
        "Right Hip",
        "Left Hip", 
        "Right Knee", 
        "Left Knee", 
        "Right Ankle"
        "Left Ankle", 
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
                cosineOfAngle = np.clip(dotProduct/magnitudesMultiplication, -1, 1)
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
                numericalConfidence, numericalPrediction = self.getPredictionRandomForest(angleSample)
            case 2:
                numericalConfidence, numericalPrediction = self.getPredictionSVM(angleSample)
            case 3:
                numericalConfidence, numericalPrediction = self.getPredictionKNN(angleSample)
            case 4:
                numericalConfidence, numericalPrediction = self.getPredictionNeuralNetwork(angleSample)

        writtenConfidence = str(int(numericalConfidence*100)) + "%"

        # Writes a prediction based on numercal prediction
        match numericalPrediction:
            case 1:
                writtenPrediction = "Good"
            case 0:
                writtenPrediction = "Bad"
            case -1:
                writtenPrediction = predictionWhenUnsure
                writtenConfidence = predictionWhenUnsure
    
        return writtenConfidence, writtenPrediction
    
    def getPredictionRandomForest(self, angleSample):
        quality = randomForest.predict([angleSample])[0]
        qualityPercentages = randomForest.predict_proba([angleSample])[0]

        confidence = max(qualityPercentages)

        #Gets quality if model is confident
        if max(qualityPercentages) > MinimumConfidenceOfPrediction:
            prediction = quality
        else:
            prediction = -1

        return confidence, prediction
    
    def getPredictionSVM(self, angleSample):
        quality = SVM.predict([angleSample])[0]
        qualityPercentages = SVM.predict_proba([angleSample])[0]

        confidence = max(qualityPercentages)
        
        #Gets quality if model is confident
        if confidence > MinimumConfidenceOfPrediction:
            prediction = quality
        else:
            prediction = -1

        return confidence, prediction
    
    def getPredictionKNN(self, angleSample):
        
        quality = KNN.predict([angleSample])[0]
        qualityPercentages = KNN.predict_proba([angleSample])[0]

        confidence = max(qualityPercentages)

        #Gets quality if model is confident
        if confidence > MinimumConfidenceOfPrediction:
            prediction = quality
        else:
            prediction = -1

        return confidence, prediction
    
    def getPredictionNeuralNetwork(self, angleSample):
        
        angleSample = np.array(angleSample).reshape(1, -1)

        qualityPercentages = NN.predict(angleSample)
        quality = tf.argmax(qualityPercentages, axis=-1).numpy()
        
        confidence = max(qualityPercentages[0])
        #Gets quality if model is confident
        if confidence > MinimumConfidenceOfPrediction:
            prediction = quality
        else:
            prediction = -1

        return confidence, prediction
    
detector = PoseDetector()
predictor = ClassPredictor()

prediction = predictionWhenUnsure
confidence = predictionWhenUnsure

countdownforModelPrediction = secondsBetweenModelPredictions

armAngleLeftArray = [0,0]
armAngleRightArray = [0,0]
angles = [0,0]
timeArray = [0,0]
repsArray =[0,0]
peaksArray = [0,0]
fps = 0
secPerRep = 0

def mainFunction(frame: av.VideoFrame):

    with lock:
        armAngleRightArray = lockContainer["armAngleRightArray"]
        armAngleLeftArray = lockContainer["armAngleLeftArray"]
        angles = lockContainer["angles"]
        timeArray = lockContainer["timeArray"]
        fps = lockContainer["fps"]
        repsArray = lockContainer["repsArray"]
        peaksArray = lockContainer["peaksArray"]
        secPerRep = lockContainer["secPerRep"]
        prediction = lockContainer["prediction"]
        confidence = lockContainer["confidence"]

    img = frame.to_ndarray(format="bgr24")
    img = cv.flip(img, 1)

    frame_width = len(img)
    frame_height = int(img.size/frame_width)

    global fpsinitTime
    fpsfinalTime = time.time()
    fps = 1 / (fpsfinalTime - fpsinitTime)
    fpsinitTime = fpsfinalTime
    timeArray.append(fpsfinalTime - initTime)

    img, armAngleRight, armAngleLeft = detector.getAngles(img, frame_width, frame_height, armAngleRightArray, armAngleLeftArray)
    
    #flips angles so that contractions are on top
    armAngleRight = 180 - armAngleRight
    armAngleLeft = 180 - armAngleLeft

    armAngleRightArray.append(int(armAngleRight))
    armAngleLeftArray.append(int(armAngleLeft))

    if rightOrLeft == "right":
            angles = armAngleRightArray
    elif rightOrLeft == "left":
            angles = armAngleLeftArray

    smoothedAngles = uniform_filter1d(angles, size = movingAverageSize)

    flippedSmoothedAngles = np.flip(smoothedAngles) #smoothedAngles is flipped so that circle of rep can be drawn easier

    peaks, _ = find_peaks(flippedSmoothedAngles, height=180-minimumAngle, distance=max(1, minimumTimeBetweenReps*fps), prominence=minimumProminenceOfReps) 

    reps = len(peaks)

    repsArray.append(reps)

    peaksArray.append(-100) #appends a low number so that the circles of non-peaks are not in the chart
    if repsArray[-1] > repsArray[-2]:
        amountOfFramesBackThatThePeakWasAt = peaks[0]
        peaksArray[-amountOfFramesBackThatThePeakWasAt] = angles[-amountOfFramesBackThatThePeakWasAt]
    
    # Guesses quality based on last 100 angles
    #if len(angles) > 110:
        #prediction = predictor.getPrediction(angle[-100:])

    #calculates seconds per rep
    if len(angles) > 110:
        try:
            secPerRep = max(0,(timeArray[-1] - timeArray[-100])/(repsArray[-1] - repsArray[-100]))
        except:
            secPerRep = 0

    # Guesses quality based on last 50 angles
    if len(angles)  > 60:
        confidence, prediction = predictor.getPrediction(angles[-50:])

    with lock:
        lockContainer["armAngleRightArray"] = armAngleRightArray
        lockContainer["armAngleLeftArray"] = armAngleLeftArray
        lockContainer["angles"] = angles
        lockContainer["timeArray"] = timeArray
        lockContainer["fps"] = fps
        lockContainer["repsArray"] = repsArray
        lockContainer["peaksArray"] = peaksArray
        lockContainer["secPerRep"] = secPerRep
        lockContainer["prediction"] = prediction
        lockContainer["confidence"] = confidence

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def endingFunction():
    with lock:
        timeArray = lockContainer["timeArray"]
        angles = lockContainer["angles"]
        peaksArray = lockContainer["peaksArray"]

    if len(timeArray) > 100:
        st.session_state.plot = "Plot is saved"
        st.session_state.timeArray = timeArray
        st.session_state.angles = angles
        st.session_state.peaksArray = peaksArray

def stream_data(string):
    for word in string.split(" "):
        time.sleep(0.07)
        yield word + " "
#==================Streamlit==================

#sets up a lock so that variables in the streamer thread can be called by outside functions
lock = threading.Lock()
lockContainer = {
    "armAngleRightArray": armAngleRightArray,
    "armAngleLeftArray": armAngleLeftArray,
    "angles": angles,
    "timeArray": timeArray,
    "fps": fps,
    "repsArray": repsArray,
    "peaksArray": peaksArray,
    "secPerRep": secPerRep,
    "prediction": prediction,
    "confidence": confidence,
    }

st.title("AI-Powered Movement Tracking for Rehabilitation")

initTime = time.time()
fpsinitTime = initTime

webrtc_ctx = webrtc_streamer(
    key="streamer",
    video_frame_callback=mainFunction,
    media_stream_constraints={"video": True, "audio": False}, 
    on_video_ended=endingFunction
    )

dataColumn, chartColumn = st.columns([1, 3])

with dataColumn:
    repsCell = st.empty()
    secPerRepCell = st.empty()

with st.sidebar:
    predictionCell = st.chat_message("assistant")

with chartColumn:
    chartCell = st.empty()

repsCell._skeleton()
secPerRepCell._skeleton()
chartCell._skeleton(height=300)
    
while webrtc_ctx.state.playing and (time.time() - initTime) < 100:

    time.sleep(0.1) #Do not remove, you will die

    with lock:
        timeArray = lockContainer["timeArray"]
        angles = lockContainer["angles"]
        peaksArray = lockContainer["peaksArray"]
        repsArray = lockContainer["repsArray"]
        prediction = lockContainer["prediction"]
        confidence = lockContainer["confidence"]
        secPerRep = lockContainer["secPerRep"]

    if len(timeArray) <= 100:
        continue

    source = pd.DataFrame({
        'Time': timeArray[-100:],
        'Angle': angles[-100:],
        'Peaks': peaksArray[-100:],
    })

    line = alt.Chart(source).mark_line().encode(
        x=alt.X('Time', axis=alt.Axis(labels=False, ticks=False, title=None), scale=alt.Scale(type='linear', domain=[timeArray[-100], timeArray[-1]])),
        y=alt.Y('Angle', axis=alt.Axis(labels=False, ticks=False, title=None), scale=alt.Scale(type='linear', domain=[0, 180]))
    )

    points = alt.Chart(source).mark_circle(
        clip=True,
        color="red",
        size=300
    ).encode(
        x=alt.X('Time', axis=alt.Axis(labels=False, ticks=False, title=None), scale=alt.Scale(type='linear', domain=[timeArray[-100], timeArray[-1]])),
        y=alt.Y('Peaks', axis=alt.Axis(labels=False, ticks=False, title=None), scale=alt.Scale(type='linear', domain=[0, 180]))
    )

    altChart = line + points

    chartCell.altair_chart(altChart)

    repsCell.write("Reps: " + str(repsArray[-1]))
    secPerRepCell.write("Seconds per Rep: " + str(int(secPerRep)))

    countdownforModelPrediction = countdownforModelPrediction - (timeArray[-1] - timeArray[-2])

    if countdownforModelPrediction <= 0:
        countdownforModelPrediction = secondsBetweenModelPredictions
        predictionCell.empty()
        predictionCell.write_stream(stream_data("I predict that you are doing a **" + str(prediction) + "** job. " + (":smiley:" if prediction == "Good" else ":worried:") + " I am **" + str(confidence) + "** confident."))

if 'plot' not in st.session_state or st.session_state.plot == 'No plot yet':
    st.session_state.plot = 'No plot yet'
else:
    print("entered new plot")

    source = pd.DataFrame({
        'Time': st.session_state.timeArray,
        'Angle': st.session_state.angles,
        'Peaks': st.session_state.peaksArray,
    })

    line = alt.Chart(source).mark_line().encode(
        x=alt.X('Time', axis=alt.Axis(labels=False, ticks=False, title=None)),
        y=alt.Y('Angle', axis=alt.Axis(labels=False, ticks=False, title=None))
    )

    points = alt.Chart(source).mark_circle(
        clip=True,
        color="red",
        size=300
    ).encode(
        x=alt.X('Time', axis=alt.Axis(labels=False, ticks=False, title=None)),
        y=alt.Y('Peaks', axis=alt.Axis(labels=False, ticks=False, title=None))
    )

    altChart = line + points

    chartCell.altair_chart(altChart)