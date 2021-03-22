# import Symbols in Engineering Drawings 
import numpy as np
import pandas as pd 

data = pd.read_csv("SiED.csv").values

train_data = data[:,:10000]
train_labels = data[:,10000]
    
#normalise data 
train_data = train_data.reshape((2431, 100 , 100, 1))
train_data= train_data.astype('float32') / 255.0

# preproessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# encoding data
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)


# Setting aside a validation set TODO CREATE VALIDATION SET 
 
val_data = train_data[0::9].copy() # start:stop:step

val_labels = train_labels[0::9].copy()

# Model definition AlexNet architecture
from keras import models 
from keras import layers 

model = models.Sequential()
model.add(layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4,4), activation='relu', input_shape=(100, 100, 1)))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3, 3)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1,1), activation='relu', padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1,1), activation='relu', padding="same"))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(39, activation='softmax'))

    
# Compilling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics= ['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

