from keras.models import load_model 
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt 


# load model
model = load_model('D:/DDocuments/Python/Practical_computer_vision/SiED_CNN_3.h5')

# Compilling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics= ['acc'])


# load the image and define the window width and height
image = cv2.imread("'D:/DDocuments/Python/Practical_computer_vision/P&ID plan.png'")    
(winW, winH) = (100, 100) 
      
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


			
            
            
def get_prediction(img_path, threshold):
  img = Image.open(img_path) # Load the image
  #img = img.convert("1")
  img = np.array(img)
  img = img.reshape((-1, 1010, 1920, 1))
  img = img.astype('float32') / 255.0
  
  
  
  pred = model([img]) # Pass the image to the model
  pred_class = [SiED_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class




def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):

  boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
  img = cv2.imread(img_path) # Read image with cv2
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()
  plt.show()