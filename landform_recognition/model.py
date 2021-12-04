# -*- coding: utf-8 -*-
#import pickle
from tensorflow import keras
import cv2
import tensorflow as tf
import numpy as np
path='C:/Users/vedant kathe/landform_recognition/test_images/10012.jpg'
img=cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (100,100)) # Resize the images
img = np.array(img) 
img = img.astype(np.int64)
img = img/255
img=np.array(img).reshape(-1,100,100,3)
reconstructed_model = keras.models.load_model("model.h5")

def predict_class(img):
    # Resize
    img = img.reshape(1,100,100,3)
    # Predict
    predictions = reconstructed_model.predict(img)
    true_prediction = [tf.argmax(pred) for pred in predictions]
    true_prediction = np.array(true_prediction)
    return print(true_prediction)

predict_class(img)