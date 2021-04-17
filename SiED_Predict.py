
from keras.models import load_model

# load model
model = load_model('D:/DDocuments/Python/Practical_computer_vision/SiED_CNN_3.h5')

# Compilling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics= ['acc'])
  

# define class names
SiED_INSTANCE_CATEGORY_NAMES = ['Arrowhead', 'Arrowhead + Triangle', 'Barred Tee', 'continuity Label',
                                'Control', 'Control Valve', 'Control Valve Angle Choke', 'Control Valve Globe',
                                'DB&BBV', 'DB&BBV + Valve Check', 'DB&BPV', 'Deluge', ' ESDV Valve Ball', 
                                'ESDV Valve Butterfly', 'ESDV Valve Slab Gate', 'Exit to Atmosphere', 
                                'Flange + Triangle', 'Flange joint', 'Flange Single T-Shape', 'Injector Point',
                                'Line Blindspacer', 'Reducer', 'Rupture Disc', 'Sensor', ' Spectacle Blind',
                                'Temporary Strainer', 'Triangle', 'Ultrasonic Flow Meter', 'Valve', 'Valve Angle'
                                'Valve Ball', 'Valve Butterfly', 'Valve Check', 'Valve Gate Through Conduit',
                                'Valve Globe', 'Valve Plug', 'Valve Slab Gate', 'Vessel']


from PIL import Image
image = 'D:/DDocuments/Python/Practical_computer_vision/P&ID plan.png'

# Global variables
THRESHOLD = 0.3
SYMBOLS = [] # an empty list for Symbol_info class

# Sliding window size & Image size
(SwinW, SwinH) = (100, 100) 
(imgW, imgH) = (1010, 1920)

     
        
import numpy as np  
from Symbol_classes import Symbol_info   # own class in Symbol_classes.py

def predict(image):
    for x in range(0,1920-100,10):
        for y in range(0, 1010-100, 10):
            img = image.crop((x, y, x+SwinW, y+SwinH)) # crop to 100x100 to fit into model
            img = img.convert('L')# converts image to a single channel
            img = np.array(img)
            img = img.reshape((-1, 100, 100, 1))
            img = img.astype('float32') / 255.0
            
            pred = model(img) # get prediction scores
            high_score = np.argmax(pred)    # get highest prediction score
            newSymbol = Symbol_info(x,y, SiED_INSTANCE_CATEGORY_NAMES[high_score], high_score)# Get the class name
            if pred[0,high_score] > THRESHOLD:
                SYMBOLS.append(newSymbol)

import cv2
import matplotlib.pyplot as plt   
import tensorflow as tf

# non-maximum suppression
boxes = []
scores = []

def nms():
    for i in range(len(SYMBOLS)):
        boxes.append(SYMBOLS[i].box)
        scores.append(SYMBOLS[i].score)

    tf.convert_to_tensor(boxes,dtype=tf.float32)
    tf.convert_to_tensor(scores,dtype=tf.float32)

    selected_indices = tf.image.non_max_suppression(boxes, scores, 20, iou_threshold=0.5)
    selected_boxes = tf.gather(boxes, selected_indices)
    
    return selected_boxes


def symbol_detection(image):        
    final_image = cv2.imread(image) # Read image with cv2
    image = Image.open(image) # Load the image
    predict(image)
    select_boxes = nms()
    tf.dtypes.cast(boxes,tf.int32)
    for i in range(len(select_boxes)):
        cv2.rectangle(final_image, (select_boxes[i][0],select_boxes[i][1]),(select_boxes[i][2],select_boxes[i][3]),color=(0, 255, 0), thickness=2) # Draw Rectangle with the coordinates
        #cv2.putText(final_image,SYMBOLS[i].label, SYMBOLS[i].box[0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2) # Write the prediction class
    plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(final_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.show()         
           
symbol_detection(image)      


