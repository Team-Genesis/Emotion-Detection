import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer-colab.json", "r").read())
#load weights
model.load_weights('fer-colab.h5')


#face_cascade = cv2.CascadeClassifier('/home/ishant/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

test_img = cv2.imread('test-angry.jpg', 0)



faces_detected = face_cascade.detectMultiScale(test_img, 1.32, 5)



for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    roi_gray=test_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
    roi_gray=cv2.resize(roi_gray,(48,48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

        #find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    print(predicted_emotion)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow('Facial emotion analysis ',resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows