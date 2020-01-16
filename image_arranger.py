#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:27:55 2020

@author: ishant
"""

import shutil, os
n = 38
files = [None] * n
for i in range (0, n-1):
    if i < 9:
        files[i] = "data/angry_facesdb/facesdb/s00" + str(i+1) + "/tif/s00" + str(i+1) + "-00_img.tif"
    else:
        files[i] = "data/angry_facesdb/facesdb/s0" + str(i+1) + "/tif/s0" + str(i+1) + "-00_img.tif"
        
#print(files)

        
for f in files:
    shutil.copy(f, "data/humans/angry")