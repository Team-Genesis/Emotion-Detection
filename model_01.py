#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:17:58 2020

@author: ishant dahiya
"""

#Model to predict human face emotions


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix

###############################
##ANGRY EMOTION################
###############################

human_angry = glob.glob("data/humans/angry/*")

print("Number of images in Angry emotion = " + str(len(human_angry)))

human_angry_folderName = ["data/humans/angry/"] * len(human_angry)
human_angry_imageName = [None] * len(human_angry)
for i in range(0, len(human_angry)):
    angry_name = human_angry[i].replace('data/humans/angry/', '')
    human_angry_imageName[i] = angry_name

human_angry_emotion = ["Angry"] * len(human_angry)
human_angry_label = [1] * len(human_angry)

df_angry = pd.DataFrame()
df_angry["folderName"] = human_angry_folderName
df_angry["imageName"] = human_angry_imageName
df_angry["Emotion"] = human_angry_emotion
df_angry["Labels"] = human_angry_label
df_angry.head()

###############################
##DISGUST EMOTION##############
###############################

human_disgust = glob.glob("data/humans/disgust/*")

print("Number of images in Disgust emotion = " + str(len(human_disgust)))

human_disgust_folderName = ["data/humans/disgust/"] * len(human_disgust)
human_disgust_imageName = [None] * len(human_disgust)
for i in range(0, len(human_disgust)):
    disgust_name = human_disgust[i].replace('data/humans/disgust/', '')
    human_disgust_imageName[i] = disgust_name

human_disgust_emotion = ["disgust"] * len(human_disgust)
human_disgust_label = [2] * len(human_disgust)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = human_disgust_folderName
df_disgust["imageName"] = human_disgust_imageName
df_disgust["Emotion"] = human_disgust_emotion
df_disgust["Labels"] = human_disgust_label
df_disgust.head()



###############################
##FEAR EMOTION##############
###############################

human_fear = glob.glob("data/humans/fear/*")

print("Number of images in fear emotion = " + str(len(human_fear)))

human_fear_folderName = ["data/humans/fear/"] * len(human_fear)
human_fear_imageName = [None] * len(human_fear)
for i in range(0, len(human_fear)):
    fear_name = human_fear[i].replace('data/humans/fear/', '')
    human_fear_imageName[i] = fear_name

human_fear_emotion = ["fear"] * len(human_fear)
human_fear_label = [3] * len(human_fear)

df_fear = pd.DataFrame()
df_fear["folderName"] = human_fear_folderName
df_fear["imageName"] = human_fear_imageName
df_fear["Emotion"] = human_fear_emotion
df_fear["Labels"] = human_fear_label
df_fear.head()

###############################
##happy EMOTION##############
###############################

human_happy = glob.glob("data/humans/happy/*")

print("Number of images in happy emotion = " + str(len(human_happy)))

human_happy_folderName = ["data/humans/happy/"] * len(human_happy)
human_happy_imageName = [None] * len(human_happy)
for i in range(0, len(human_happy)):
    happy_name = human_happy[i].replace('data/humans/happy/', '')
    human_happy_imageName[i] = happy_name

human_happy_emotion = ["happy"] * len(human_happy)
human_happy_label = [4] * len(human_happy)

df_happy = pd.DataFrame()
df_happy["folderName"] = human_happy_folderName
df_happy["imageName"] = human_happy_imageName
df_happy["Emotion"] = human_happy_emotion
df_happy["Labels"] = human_happy_label
df_happy.head()

###############################
##neutral EMOTION##############
###############################

human_neutral = glob.glob("data/humans/neutral/*")

print("Number of images in neutral emotion = " + str(len(human_neutral)))

human_neutral_folderName = ["data/humans/neutral/"] * len(human_neutral)
human_neutral_imageName = [None] * len(human_neutral)
for i in range(0, len(human_neutral)):
    neutral_name = human_neutral[i].replace('data/humans/neutral/', '')
    human_neutral_imageName[i] = neutral_name

human_neutral_emotion = ["neutral"] * len(human_neutral)
human_neutral_label = [5] * len(human_neutral)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = human_neutral_folderName
df_neutral["imageName"] = human_neutral_imageName
df_neutral["Emotion"] = human_neutral_emotion
df_neutral["Labels"] = human_neutral_label
df_neutral.head()

###############################
##sad EMOTION##############
###############################

human_sad = glob.glob("data/humans/sad/*")

print("Number of images in sad emotion = " + str(len(human_sad)))

human_sad_folderName = ["data/humans/sad/"] * len(human_sad)
human_sad_imageName = [None] * len(human_sad)
for i in range(0, len(human_sad)):
    sad_name = human_sad[i].replace('data/humans/sad/', '')
    human_sad_imageName[i] = sad_name

human_sad_emotion = ["sad"] * len(human_sad)
human_sad_label = [6] * len(human_sad)

df_sad = pd.DataFrame()
df_sad["folderName"] = human_sad_folderName
df_sad["imageName"] = human_sad_imageName
df_sad["Emotion"] = human_sad_emotion
df_sad["Labels"] = human_sad_label
df_sad.head()

frames = [df_angry, df_disgust, df_fear, df_happy, df_neutral, df_sad]
final_human = pd.concat(frames)
final_human.shape

df_human_train_data, df_human_test = train_test_split(final_human, stratify=final_human["Labels"], test_size = 0.197860)
df_human_train, df_human_cv = train_test_split(df_human_train_data, stratify=df_human_train_data["Labels"], test_size = 0.166666)
df_human_train.shape, df_human_cv.shape, df_human_test.shape


df_human_train.reset_index(inplace = True, drop = True)
df_human_train.to_pickle("data/dataframes/human/df_human_train.pkl")

df_human_cv.reset_index(inplace = True, drop = True)
df_human_cv.to_pickle("data/dataframes/human/df_human_cv.pkl")

df_human_test.reset_index(inplace = True, drop = True)
df_human_test.to_pickle("data/dataframes/human/df_human_test.pkl")

df_human_train = pd.read_pickle("data/dataframes/human/df_human_train.pkl")
df_human_train.head()

df_human_train.shape

df_human_cv = pd.read_pickle("data/dataframes/human/df_human_cv.pkl")
df_human_cv.head()

df_human_cv.shape

df_human_test = pd.read_pickle("data/dataframes/human/df_human_test.pkl")
df_human_test.head()

df_human_test.shape

#Analysing Data of Human Images¶

#Distribution of class labels in Train, CV and Test

df_temp_train = df_human_train.sort_values(by = "Labels", inplace = False)
df_temp_cv = df_human_cv.sort_values(by = "Labels", inplace = False)
df_temp_test = df_human_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_human_train["Emotion"].value_counts().sort_index()
CVData_distribution = df_human_cv["Emotion"].value_counts().sort_index()
TestData_distribution = df_human_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
CVData_distribution_sorted = sorted(CVData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)


fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.2, y = i.get_height()+1.5, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Validation Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_cv)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in CVData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_cv.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")

#Pre processing available Data Image to bw
    
def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))
    
convt_to_gray(df_human_train)
convt_to_gray(df_human_cv)
convt_to_gray(df_human_test)

#detect the face in image using HAAR cascade then crop it then resize it and finally save it.
face_cascade = cv2.CascadeClassifier('/home/ishant/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml') 
#download this xml file from link: https://github.com/opencv/opencv/tree/master/data/haarcascades.
def face_det_crop_resize(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]  #cropping the face in image
        cv2.imwrite(img_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it


for i, d in df_human_train.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)


for i, d in df_human_cv.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)
    
for i, d in df_human_test.iterrows():
    img_path = os.path.join(d["folderName"], d["imageName"])
    face_det_crop_resize(img_path)
    
    
####################################
    #USING SCRAPED IMAGES############
    #################################

scraped_angry = glob.glob("data/scraper/angry/*")

print("Number of images in Angry emotion = " + str(len(scraped_angry)))

scraped_angry_folderName = ["data/scraper/angry/"] * len(scraped_angry)
scraped_angry_imageName = [None] * len(scraped_angry)
for i in range(0, len(scraped_angry)):
    angry_name = scraped_angry[i].replace('data/scraper/angry/', '')
    scraped_angry_imageName[i] = angry_name

scraped_angry_emotion = ["Angry"] * len(scraped_angry)
scraped_angry_label = [1] * len(scraped_angry)

df_angry = pd.DataFrame()
df_angry["folderName"] = scraped_angry_folderName
df_angry["imageName"] = scraped_angry_imageName
df_angry["Emotion"] = scraped_angry_emotion
df_angry["Labels"] = scraped_angry_label
df_angry.head()

###############################
##DISGUST EMOTION##############
###############################

scraped_disgust = glob.glob("data/scraper/disgust/*")

print("Number of images in disgust emotion = " + str(len(scraped_disgust)))

scraped_disgust_folderName = ["data/scraper/disgust/"] * len(scraped_disgust)
scraped_disgust_imageName = [None] * len(scraped_disgust)
for i in range(0, len(scraped_disgust)):
    disgust_name = scraped_disgust[i].replace('data/scraper/disgust/', '')
    scraped_disgust_imageName[i] = disgust_name

scraped_disgust_emotion = ["disgust"] * len(scraped_disgust)
scraped_disgust_label = [2] * len(scraped_disgust)

df_disgust = pd.DataFrame()
df_disgust["folderName"] = scraped_disgust_folderName
df_disgust["imageName"] = scraped_disgust_imageName
df_disgust["Emotion"] = scraped_disgust_emotion
df_disgust["Labels"] = scraped_disgust_label
df_disgust.head()



###############################
##FEAR EMOTION##############
###############################

scraped_fear = glob.glob("data/scraper/fear/*")

print("Number of images in fear emotion = " + str(len(scraped_fear)))

scraped_fear_folderName = ["data/scraper/fear/"] * len(scraped_fear)
scraped_fear_imageName = [None] * len(scraped_fear)
for i in range(0, len(scraped_fear)):
    fear_name = scraped_fear[i].replace('data/scraper/fear/', '')
    scraped_fear_imageName[i] = fear_name

scraped_fear_emotion = ["fear"] * len(scraped_fear)
scraped_fear_label = [3] * len(scraped_fear)

df_fear = pd.DataFrame()
df_fear["folderName"] = scraped_fear_folderName
df_fear["imageName"] = scraped_fear_imageName
df_fear["Emotion"] = scraped_fear_emotion
df_fear["Labels"] = scraped_fear_label
df_fear.head()

###############################
##happy EMOTION##############
###############################

scraped_happy = glob.glob("data/scraper/happy/*")

print("Number of images in happy emotion = " + str(len(scraped_happy)))

scraped_happy_folderName = ["data/scraper/happy/"] * len(scraped_happy)
scraped_happy_imageName = [None] * len(scraped_happy)
for i in range(0, len(scraped_happy)):
    happy_name = scraped_happy[i].replace('data/scraper/happy/', '')
    scraped_happy_imageName[i] = happy_name

scraped_happy_emotion = ["happy"] * len(scraped_happy)
scraped_happy_label = [4] * len(scraped_happy)

df_happy = pd.DataFrame()
df_happy["folderName"] = scraped_happy_folderName
df_happy["imageName"] = scraped_happy_imageName
df_happy["Emotion"] = scraped_happy_emotion
df_happy["Labels"] = scraped_happy_label
df_happy.head()

###############################
##neutral EMOTION##############
###############################

scraped_neutral = glob.glob("data/scraper/neutral/*")

print("Number of images in neutral emotion = " + str(len(scraped_neutral)))

scraped_neutral_folderName = ["data/scraper/neutral/"] * len(scraped_neutral)
scraped_neutral_imageName = [None] * len(scraped_neutral)
for i in range(0, len(scraped_neutral)):
    neutral_name = scraped_neutral[i].replace('data/scraper/neutral/', '')
    scraped_neutral_imageName[i] = neutral_name

scraped_neutral_emotion = ["neutral"] * len(scraped_neutral)
scraped_neutral_label = [5] * len(scraped_neutral)

df_neutral = pd.DataFrame()
df_neutral["folderName"] = scraped_neutral_folderName
df_neutral["imageName"] = scraped_neutral_imageName
df_neutral["Emotion"] = scraped_neutral_emotion
df_neutral["Labels"] = scraped_neutral_label
df_neutral.head()

###############################
##sad EMOTION##############
###############################

scraped_sad = glob.glob("data/scraper/sad/*")

print("Number of images in sad emotion = " + str(len(scraped_sad)))

scraped_sad_folderName = ["data/scraper/sad/"] * len(scraped_sad)
scraped_sad_imageName = [None] * len(scraped_sad)
for i in range(0, len(scraped_sad)):
    sad_name = scraped_sad[i].replace('data/scraper/sad/', '')
    scraped_sad_imageName[i] = sad_name

scraped_sad_emotion = ["sad"] * len(scraped_sad)
scraped_sad_label = [1] * len(scraped_sad)

df_sad = pd.DataFrame()
df_sad["folderName"] = scraped_sad_folderName
df_sad["imageName"] = scraped_sad_imageName
df_sad["Emotion"] = scraped_sad_emotion
df_sad["Labels"] = scraped_sad_label
df_sad.head()

frames = [df_angry, df_disgust, df_fear, df_happy, df_neutral, df_sad]
final_scraped = pd.concat(frames)
final_scraped.shape

#Train Test CV Data for Scraped Images

df_scraped_train_data, df_scraped_test = train_test_split(final_scraped, stratify=final_scraped["Labels"], test_size = 0.181868)
df_scraped_train, df_scraped_cv = train_test_split(df_scraped_train_data, stratify=df_scraped_train_data["Labels"], test_size = 0.148607)
df_scraped_train.shape, df_scraped_cv.shape, df_scraped_test.shape

df_scraped_train.reset_index(inplace = True, drop = True)
df_scraped_train.to_pickle("data/dataframes/scraper/df_scraped_train.pkl")

df_scraped_cv.reset_index(inplace = True, drop = True)
df_scraped_cv.to_pickle("data/dataframes/scraper/df_scraped_cv.pkl")

df_scraped_test.reset_index(inplace = True, drop = True)
df_scraped_test.to_pickle("data/dataframes/scraper/df_scraped_test.pkl")


df_scraped_train = pd.read_pickle("data/dataframes/scraper/df_scraped_train.pkl")
df_scraped_train.head()

df_scraped_train.shape

df_scraped_cv = pd.read_pickle("data/dataframes/scraper/df_scraped_cv.pkl")
df_scraped_cv.head()

df_scraped_cv.shape

df_scraped_test = pd.read_pickle("data/dataframes/scraper/df_scraped_test.pkl")
df_scraped_test.head()

df_scraped_test.shape


##PLOTTING THE SCRAPED DATA

df_temp_train = df_scraped_train.sort_values(by = "Labels", inplace = False)
df_temp_cv = df_scraped_cv.sort_values(by = "Labels", inplace = False)
df_temp_test = df_scraped_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_scraped_train["Emotion"].value_counts().sort_index()
CVData_distribution = df_scraped_cv["Emotion"].value_counts().sort_index()
TestData_distribution = df_scraped_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
CVData_distribution_sorted = sorted(CVData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.185, y = i.get_height()+1.6, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Validation Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_cv)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.21, y = i.get_height()+0.3, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in CVData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_cv.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.21, y = i.get_height()+0.3, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")
    
convt_to_gray(df_scraped_train)
convt_to_gray(df_scraped_cv)
convt_to_gray(df_scraped_test)



def change_image(df):
    count = 0
    for i, d in df.iterrows():
        img = cv2.imread(os.path.join(d["folderName"], d["imageName"]))
        face_clip = img[40:240, 35:225]         #cropping the face in image
        face_resized = cv2.resize(face_clip, (350, 350))
        cv2.imwrite(os.path.join(d["folderName"], d["imageName"]), face_resized) #resizing and saving the image
        count += 1
    print("Total number of images cropped and resized = {}".format(count))

change_image(df_scraped_train)
change_image(df_scraped_cv)
change_image(df_scraped_test)
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    