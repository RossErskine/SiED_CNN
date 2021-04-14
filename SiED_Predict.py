from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt 
from Symbol_classes import Symbol_info, Lines

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

image = 'D:/DDocuments/Python/Practical_computer_vision/P&ID plan.png'

THRESHOLD = 0.3
SYMBOLS = []

def is_unique(symbol):
    flag = True 
    for s in SYMBOLS:
        if symbol.box[0][0] > s.box[0][0] -50 and symbol.box[0][0] < s.box[0][0] +50: # symbol x betwwen 50 0f s x
            if symbol.box[0][1] > s.box[0][1] +50 and symbol.box[0][1] < s.box[0][1] +50 :# symbol y betwwen 50 0f s y
                if symbol.label == s.label:
                    if symbol.score < s.score:
                        flag = False
                        return False
                    else:
                        del s
                        flag = True
                        
    if flag == True:
        return True
    
     
        
    
    

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
                if is_unique(newSymbol):
                    SYMBOLS.append(newSymbol)
 
def symbol_detection(image):        
    final_image = cv2.imread(image) # Read image with cv2
    image = Image.open(image) # Load the image
    predict(image)
    for i in range(len(SYMBOLS)):
        cv2.rectangle(final_image, SYMBOLS[i].box[0], SYMBOLS[i].box[1],color=(0, 255, 0), thickness=2) # Draw Rectangle with the coordinates
        cv2.putText(final_image,SYMBOLS[i].label, SYMBOLS[i].box[0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2) # Write the prediction class
    plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(final_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.show()         
           
symbol_detection(image)      


