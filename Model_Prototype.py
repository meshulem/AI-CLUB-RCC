#preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#processing (4 models)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

#postprocessing
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

CSV = pd.read_csv('Training_Protype_New.csv')

X = CSV[["Angle: 1","Angle: 2","Angle: 3","Angle: 4","Angle: 5","Angle: 6","Angle: 7","Angle: 8","Angle: 9","Angle: 10","Angle: 11","Angle: 12","Angle: 13","Angle: 14","Angle: 15","Angle: 16","Angle: 17","Angle: 18","Angle: 19","Angle: 20","Angle: 21","Angle: 22","Angle: 23","Angle: 24","Angle: 25","Angle: 26","Angle: 27","Angle: 28","Angle: 29","Angle: 30","Angle: 31","Angle: 32","Angle: 33","Angle: 34","Angle: 35","Angle: 36","Angle: 37","Angle: 38","Angle: 39","Angle: 40","Angle: 41","Angle: 42","Angle: 43","Angle: 44","Angle: 45","Angle: 46","Angle: 47","Angle: 48","Angle: 49","Angle: 50"]]
y = CSV['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = X_train/180
X_test_scaled = X_test/180

print("====================RandomForestClassifier=======================")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred_RF = rf_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_RF)
classification_rep = classification_report(y_test, y_pred_RF)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

joblib.dump(rf_classifier, "Saved_Model_Prototype_RF.joblib")

print("=================Support Vector Machine(SVM)=================")

svm_classifier = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
svm_classifier.fit(X_train_scaled, y_train)

y_pred_SVM = svm_classifier.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_SVM):.2f}")
print(classification_report(y_test, y_pred_SVM))

joblib.dump(svm_classifier, "Saved_Model_Prototype_SVM.joblib")

print("=================K-Nearest Neighbors (KNN)=================")

knn_classifier = KNeighborsClassifier(n_neighbors=5 )
knn_classifier.fit(X_train_scaled, y_train)

y_pred_KNN = knn_classifier.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_KNN):.2f}")
print(classification_report(y_test, y_pred_KNN))

joblib.dump(knn_classifier, "Saved_Model_Prototype_KNN.joblib")

print("=====================Neural Network (NN)=====================")

#Converts class numbers to one-hot encoded vectors (see https://en.wikipedia.org/wiki/One-hot)
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_test_onehot = tf.keras.utils.to_categorical(y_test)
 
# Defining Model
# Using Sequential() to build layers one after another
NN_classifier = tf.keras.Sequential([
 
  # Flatten Layer that converts images to 1D array
  tf.keras.layers.Flatten(),
   
  # Hidden Layer with 512 units and relu activation
  tf.keras.layers.Dense(units=100, activation='relu'),
   
  # Output Layer with 10 units for 10 classes and softmax activation
  tf.keras.layers.Dense(units=2, activation='softmax')
])
 
NN_classifier.compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = ['accuracy']
)
 
history = NN_classifier.fit(
  x = X_train_scaled,
  y = y_train_onehot,
  epochs = 10
)

predicted_probabilities = NN_classifier.predict(X_test_scaled)
y_pred_NN = tf.argmax(predicted_probabilities, axis=-1).numpy()
print(f"Accuracy: {accuracy_score(y_test, y_pred_NN):.2f}")
print(classification_report(y_test, y_pred_NN))

joblib.dump(NN_classifier, "Saved_Model_Prototype_NN.joblib")

print("======================The End=======================")