import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import os

#load model
model = model_from_json(open("fer-colab30.json", "r").read())
#load weights
model.load_weights('fer-colab30.h5')


#face_cascade = cv2.CascadeClassifier('/home/ishant/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict_save(test_img, save_path, suffix):
    faces_detected = face_cascade.detectMultiScale(test_img, 1.32, 5)
    predicted_emotion = ''
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
        
        if predicted_emotion == 'angry' or 'disgust' or 'fear' or 'sad' :
            predicted_emotion = 'not interested'
        elif predicted_emotion == 'happy' or 'surprise':
            predicted_emotion = 'interested'
        else:
            predicted_emotion = 'neutral'
    
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imwrite(save_path + '/' + predicted_emotion + suffix + '.jpg', resized_img)
    return predicted_emotion

happy_path = 'data/scraper (copy)/happy'
angry_path = 'data/scraper (copy)/angry'
disgust_path = 'data/scraper (copy)/disgust'
neutral_path = 'data/scraper (copy)/neutral'
sad_path = 'data/scraper (copy)/sad'
emotion_paths = ['data/scraper (copy)/predicted/happy', 'data/scraper (copy)/predicted/angry', 'data/scraper (copy)/predicted/disgust', 'data/scraper (copy)/predicted/neutral', 'data/scraper (copy)/predicted/sad']
emotions = ['happy', 'angry', 'disgust', 'neutral', 'sad' ]

files = [[] for i in range(5)]
accuracy = [0] * 5


def get_images(path, n):
    global files
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files[n].append(os.path.join(r, file))

get_images(happy_path, 0)
get_images(angry_path, 1)
get_images(disgust_path, 2)
get_images(neutral_path, 3)
get_images(sad_path, 4)



for i in range(0, 5):
    emotion = emotions[i]
    path = emotion_paths[i]
    
    
    for j in range(0, len(files[i])):
        
        test_img = cv2.imread(files[i][j], 0)
        
        if emotion == 'happy' :
            emotion = 'interested'
        elif emotion == 'neutral':
            emotion = 'neutral'
        else :
            emotion = 'not interested'
        
        img_emotion = predict_save(test_img, path, str(i) + '_' + str(j))
        
        print("Emotion:", emotion)
        print("Predicted Emotion:", img_emotion )
        
        if img_emotion == emotion:
            accuracy[i] = accuracy[i] + 1
            print("correct prediction. YEAHHHH!")
        
        print("total images left in this emotion: ", len(files[i]) - j)
        
for x in range(0, len(files)):
    accuracy[x] = (accuracy[x]/len(files[x])) * 100

print(accuracy)
        
    
    
    
    


    