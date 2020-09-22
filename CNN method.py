# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:37:28 2020

@author: Yamini
"""
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os
def load_dataset(img_dimension):
    data = [x.strip() for x in open('centers.txt','r').readlines()]
    all_images = np.zeros((len(data),img_dimension[0],img_dimension[1],img_dimension[2]))
    all_centers = np.zeros((len(data),3))
    index = 0
    no_of_blacklist = 0
    for row in data:
        filename = row.split(' [[[')[0].split('\\')[-1]
        session = int(filename.split('_')[2])
        folder = int(filename.split('_')[1])
        center_info = np.array([int(x) for x in row.split(' [[[')[1].replace(']','').split()])
        img_path = './UBIRIS_200_150/Sessao_{}/{}/{}'.format(session,folder,filename)
        img_data = cv2.imread(img_path)
        if img_data.shape != img_dimension:
            no_of_blacklist = no_of_blacklist + 1
            print(filename, '- Dimensionality mismatch!')
            continue
        all_images[index,:,:,:] = img_data
        all_centers[index,:] = center_info
        index = index + 1
    all_images= all_images[:-no_of_blacklist,:,:,:]
    all_centers= all_centers[:-no_of_blacklist,:]
    return all_images, all_centers



data, outputs = load_dataset((150,200,3))
train_data,  test_data,y_train, y_test = train_test_split(data,outputs)
train_data, test_data = train_data / 255.0, test_data / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))

model.compile(optimizer='adam', loss = tf.keras.losses.MeanSquaredError())

history = model.fit(train_data, y_train, epochs=10, 
                    validation_data=(test_data, y_test))

y_pred = model.predict(test_data)
if not os.path.exists('Predicted iris'):
    os.makedirs('Predicted iris')
for i in range(test_data.shape[0]):
    a = (test_data[i,:,:,:]*255).astype(np.uint8)
    c = cv2.circle(a, tuple(y_pred[i,:][:2].astype(np.uint8)),int(y_pred[i,:][2]), (0,0,255),5)
    cv2.imwrite(f"./Predicted iris/{i}.jpg",c)

#