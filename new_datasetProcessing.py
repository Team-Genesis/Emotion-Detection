#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:23:05 2020

@author: ishant
: ishant
"""

import shutil, os
n = 201
happy_files = [None] * n
neutral_files = [None] * n
for i in range (0, n-1):
    if i < 100:
        neutral_files[i] = "data/frontalimages_manuallyaligned_part1/" + str(i+1) + "a.jpg"
        happy_files[i] = "data/frontalimages_manuallyaligned_part1/" + str(i+1) + "b.jpg"
    else:
        neutral_files[i] = "data/frontalimages_manuallyaligned_part2/" + str(i+1) + "a.jpg"
        happy_files[i] = "data/frontalimages_manuallyaligned_part2/" + str(i+1) + "b.jpg"  


happy_files.remove(None)
neutral_files.remove(None)

for i in range(0, n-1):
    shutil.copy(neutral_files[i], "data/humans/neutral")
    shutil.copy(happy_files[i], "data/humans/happy")
        
        
 #shutil.copy(files[i], "data/humans/neutral")