# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:22:02 2019

@author: omkar
"""
from PIL import Image
import numpy as np
import cv2
import keras
import os
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
#from matplotlib import pyplot as plt
#from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib.pyplot as plt
test_path=[]

train_path='train'
valid_path='valid'
test_path='test'

train_batches=ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['bag','bottle','duster','laptop','mobile'], batch_size=10)
valid_batches=ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['bag','bottle','duster','laptop','mobile'], batch_size=10)
test_batches=ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['bag','bottle','duster','laptop','mobile'], batch_size=10)
vgg16_model=keras.applications.vgg16.VGG16()

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable=False

model.add(Dense(5 ,activation='softmax'))
model.summary()
model.compile(Adam(lr=.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=5,
                 validation_data=valid_batches,validation_steps=5, epochs=20, verbose=2)

model.save_weights("model_14-11.h5")

			 
'''dataset_path = "test.jpg"
classes = os.listdir(dataset_path)
x_train,y_train = [ ],[ ]
for folder_name in os.listdir(dataset_path):
    for image_name in os.listdir(dataset_path + folder_name):
        image = cv2.imread(dataset_path + folder_name + '/' + image_name )
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_array = Image.fromarray(image,'RGB')
        image_array = image_array.resize((128,128))
        y_train.append(classes.index(folder_name)) 
        x_train.append(numpy.array(image_array))
        #plt.imshow(image_array)
        #plt.show()
'''


'''
img=cv2.imread("test.jpg",0)
img=img.resize(2,(225,225),3)
print(img)
pre=model.predict(img, batch_size=None, verbose=0,steps=1)
print(pre)
'''
'''
image=[]
imagePaths = ('test')
data = []
labels = []
 
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[0]
 
	# load the image, swap color channels, and resize it to be a fixed
	# 128x128 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (225,225))
 
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
x=[]
x = img_to_array(test_batches)  
x = x.reshape((3, 224, 224))

pre=model.predict_generator(test_batches,batch_size=10,verbose=0)

'''