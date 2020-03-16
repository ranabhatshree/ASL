# -*- coding: utf-8 -*-

# import numpy as np
from keras.models import model_from_json
import cv2
import operator
import numpy as np

IMG_SIZE = 64


# Loading the model
json_file = open("ASL-model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("ASL-model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26:'space', 27:'del', 28:'nothing'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    #print(frame.shape)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) 

    cv2.imshow("output", roi)
    
    # Batch of 1
    result = loaded_model.predict(roi.reshape(1, IMG_SIZE, IMG_SIZE, 3))
    final_result = np.argmax(result)

    prediction = categories[final_result]
    
    # Displaying the predictions
    cv2.putText(frame, str(prediction), (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)    
    cv2.imshow("Real Frame", frame)
    
    
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
