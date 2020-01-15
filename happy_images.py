#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:27:55 2020

@author: ishant
"""

import shutil, os
n = 37
files = [None] * n
for i in range (0, n-1):
    if i < 9:
        files[i] = "data/happy/facesdb/s00" + str(i+1) + "/tif/s00" + str(i+1) + "-05_img.tif"
    elif i==19 or i==21:
        print("File not found")
    else:
        files[i] = "data/happy/facesdb/s0" + str(i+1) + "/tif/s0" + str(i+1) + "-05_img.tif"
        
#print(files)
print(files)
for i in range (0, n-2):
    if files[i] != None:
        print(files[i])
        shutil.copy(files[i], "data/humans/happy")