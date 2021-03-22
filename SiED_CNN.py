# import Symbols in Engineering Drawings 
import numpy as np
import pandas as pd 

data = pd.read_csv("SiED.csv").values

# Shuffle data for validation split
np.random.shuffle(data)

#slice data int data and  labels
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


# Setting aside a validation set 
val_data = train_data[2188:]
train_x = train_data[:2188]

val_labels = train_labels[2188:]
train_y = train_labels[:2188]

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
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics= ['acc'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

# Saving the model
model.save('SiED_CNN_1.h5')

# Plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plotting the trainig and validation loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the trainig and validation accuracy
plt.clf() # clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuarcy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
