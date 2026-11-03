# AI-Powered Movement Tracking System for Rehabilitation

**Goal:** Build a computer vision application that uses a standard webcam to track a person’s movements, analyze their range of motion, and give feedback during simulated rehabilitation exercises.

**Example:** Someone raises their arm from their side to shoulder height. The system will:
* Identify the shoulder, elbow, and wrist
* Track those joints as the person moves
* Calculate approximate joint angles and range of motion
* Count repetitions
* Compare the movement to a predefined reference exercise
* Identify deviations from the expected movement pattern
* Display results on an interactive dashboard

**Outline:**
1. Capture/read video with OpenCv
2. Use the following pose detection models to estimate key locations and show them above the video
   * MediaPipe
   * Movenet
   * OpenPose
3. Use vector math to calculate as much data as possible *(key joint angles, movement speed, consistency, reps completed, range of motion, etc.)*
4. Record video of exercises done and get back the data calculated in step 3
5. Label video data with the exercise performed and quality of exercise
6. Use the following machine learning models to guess exercise and evaluate quality of exercise based on video data
   * Teachable Machine
   * Random Forest
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Neural Networks
7. Use Steamlit to record video and return guessed exercise, video data, and quality of exercise. 
