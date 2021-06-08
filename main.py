# import pandas as pd
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import itertools
import keras
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential
# from keras import optimizers
# from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
# from keras.utils.np_utils import to_categorical
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import math
# import datetime
# import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

###########################
# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)
file_list = ['C:\\Users\\aksha\\PycharmProjects\\mediapipe\\images.jpg']
for idx, file in enumerate(file_list):
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  image = cv2.flip(cv2.imread(file), 1)

  # Convert the BGR image to RGB before processing.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  # Print handedness and draw hand landmarks on the image.
  ##print('handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    continue
  annotated_image = image.copy()
  for hand_landmarks in results.multi_hand_landmarks:
    ##print('hand_landmarks:', hand_landmarks)
    mp_drawing.draw_landmarks(
        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imwrite(
      '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1))
hands.close()
#############################
# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

vgg16 = applications.VGG16(include_top=False, weights='imagenet')

cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  # image = cv2.imread('C:\\Users\\aksha\\PycharmProjects\\mediapipe\\images.jpg')
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        data = []
        cars = results.multi_hand_landmarks
        for car in cars:
            for val in car.landmark:
                data.append(val.x)
        ##print(data)
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  display = image
  crop_img = image[0:300, 384:630]
  cv2.imshow("cropped", crop_img)


  # def read_image(file_path):
  #     print("[INFO] loading and preprocessing image...")
  #     image = load_img(file_path, target_size=(224, 224))
  #     image = img_to_array(image)
  #     image = np.expand_dims(image, axis=0)
  #     image /= 255.
  #     return image


  # orig = cv2.imread(image_path)

  bigger = cv2.resize(crop_img, (244, 244))

  # print("[INFO] loading and preprocessing image...")
  # image = load_img(image_path, target_size=(224, 224))
  image = img_to_array(bigger)

  # important! otherwise the predictions will be '0'
  image = image / 255

  image = np.expand_dims(image, axis=0)


  # build the VGG16 network
  model = applications.VGG16(include_top=False, weights='imagenet')

  # get the bottleneck prediction from the pre-trained VGG16 model
  bottleneck_prediction = model.predict(image)

  # build top model
  model = Sequential()
  model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
  model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
  model.add(Dropout(0.3))
  model.add(Dense(4, activation='softmax'))

  model.load_weights('bottleneck_fc_model.h5')

  # use the bottleneck prediction on the top model to get the final classification
  class_predicted = model.predict_classes(bottleneck_prediction)

  inID = class_predicted[0]

  class_dictionary = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

  inv_map = {v: k for k, v in class_dictionary.items()}

  label = inv_map[inID]

  # get the prediction label
  ##print("Image ID: {}, Label: {}".format(inID, label))


  animals = ['A', 'B', 'C', 'D']
  images = image
  # time.sleep(.5)

  bt_prediction = vgg16.predict(images)
  preds = model.predict_proba(bt_prediction)
  for idx, animal, x in zip(range(0, 4), animals, preds[0]):
      print("ID: {}, Label: {} {}%".format(idx, animal, round(x * 100, 2)))
  print('Final Decision:')
  # time.sleep(.5)
  for x in range(3):
      print('.' * (x + 1))
  # time.sleep(.2)
  class_predicted = model.predict_classes(bt_prediction)
  class_dictionary = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
  inv_map = {v: k for k, v in class_dictionary.items()}
  print("ID: {}, Label: {}".format(class_predicted[0], inv_map[class_predicted[0]]))


  # display the predictions with the image
  cv2.putText(display, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
  cv2.rectangle(display, (384, 0), (630, 300), (0, 255, 0), 3)
  # cv2.imshow("Classification", orig)
  cv2.imshow('MediaPipe Hands', display)
  if cv2.waitKey(5) & 0xFF == 27:
      break
hands.close()
cap.release()