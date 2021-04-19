
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
     
        
import numpy as np 
from Symbol_classes import Symbol_info   # own class in Symbol_classes.py


def predict(image, scale, original_image, power):
    #image = Image.open(image) # Load the image
    for x in range(0,image.size[1]-50,8):    #width
        for y in range(0, image.size[0]-50, 8):  #height
            img = image.crop((x, y, x+100, y+100)) # crop to 100x100 to fit into model
            img = img.convert('L')# converts image to a single channel
            img = np.array(img)
            img = img.reshape((-1, 100, 100, 1))
            img = img.astype('float32') / 255.0
            
            pred = model(img) # get prediction scores
            high_score_pos = np.argmax(pred)    # get highest prediction score
            newSymbol = Symbol_info(x,y, SiED_INSTANCE_CATEGORY_NAMES[high_score_pos], pred[0,high_score_pos], scale, original_image, power)# Get the class name
            if pred[0,high_score_pos] > THRESHOLD:
                SYMBOLS.append(newSymbol)

def pyramid(image, scale=1.1, minSize=(1000, 1500)):
     original_image = True
     power = 1
     while True:
        # compute the new dimensions of the image and resize it
        w = int(image.size[1] / scale)
        h = int(image.size[0] / scale)
        image = image.resize((h, w))
        predict(image, scale, original_image, power)
        original_image = False
        power += 1
        #if the resized image does not meet the supplied minimum
        #size, then stop constructing the pyramid
        if image.size[0] < minSize[1] or image.size[1] < minSize[0]:
           break
    
   

def selectSymbols(selected_indices):
    selected_symbols = []
    for i in selected_indices:
        selected_symbols.append(SYMBOLS[i])
    return selected_symbols

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

    selected_indices = tf.image.non_max_suppression(boxes, scores, 20, iou_threshold=0.1)
    selected_symbols = selectSymbols(selected_indices)
    
    return selected_symbols


def symbol_detection(image):        
    final_image = cv2.imread(image) # Read image with cv2
    #image = cv2.imread(image)
    image = Image.open(image) # Load the image
    pyramid(image) #pyramids image and predicts
    select_symbols = nms() # non-max suppresion
    for i in range(len(select_symbols)):
        cv2.rectangle(final_image, (select_symbols[i].box[0],select_symbols[i].box[1]),(select_symbols[i].box[2],select_symbols[i].box[3]),color=(0, 255, 0), thickness=2) # Draw Rectangle with the coordinates
        cv2.putText(final_image,select_symbols[i].label, (select_symbols[i].box[0],select_symbols[i].box[1]),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2) # Write the prediction class
    plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(final_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.show()         
           
symbol_detection(image)      


