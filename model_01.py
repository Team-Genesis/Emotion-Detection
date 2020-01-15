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




