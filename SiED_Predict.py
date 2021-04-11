from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt 

# load model
model = load_model('D:/DDocuments/Python/Practical_computer_vision/SiED_CNN_3.h5')

# Compilling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics= ['acc'])

image = ('D:/DDocuments/Python/Practical_computer_vision/P&ID plan.png')   

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

# Sliding window size & Image size
(SwinW, SwinH) = (100, 100) 
(imgW, imgH) = (1010, 1920)

image = Image.open(image) # Load the image
threshold = 0.3
pred_boxes =[]
pred_class = []


for x in range(0,1920-100,10):
    for y in range(0, 1010-100, 10):
        img = image.crop((x, y, x+SwinW, y+SwinH)) # crop to 100x100 to fit into model
        img = img.convert('L')# converts image to a single channel
        img = np.array(img)
        img = img.reshape((-1, 100, 100, 1))
        img = img.astype('float32') / 255.0
        
        pred = model(img)
        pred_h = np.argmax(pred)    # get prediction score
        clsf = SiED_INSTANCE_CATEGORY_NAMES[pred_h]  # Get the class name
        box = ((x, y), (x+SwinW, y+SwinH))  # Bounding boxes
        if pred[0,pred_h] > threshold:
            pred_boxes.append(box)
            pred_class.append(clsf)
        
final_image = cv2.imread('D:/DDocuments/Python/Practical_computer_vision/P&ID plan.png') # Read image with cv2
for i in range(len(pred_boxes)):
    cv2.rectangle(final_image, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=2) # Draw Rectangle with the coordinates
    cv2.putText(final_image,pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2) # Write the prediction class
plt.figure(figsize=(20,30)) # display the output image
plt.imshow(final_image)
plt.xticks([])
plt.yticks([])
plt.show()
plt.show()         
           
         


