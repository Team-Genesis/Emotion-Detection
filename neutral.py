#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:08:10 2020

@author: ishant
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:27:55 2020

@author: ishant
"""

import shutil
n = 37
files = [None] * n
for i in range (0, n-1):
    if i < 9:
        files[i] = "data/sad/facesdb/s00" + str(i+1) + "/tif/s00" + str(i+1) + "-02_img.tif"
    elif i==19 or i==21:
        print("File not found")
    else:
        files[i] = "data/sad/facesdb/s0" + str(i+1) + "/tif/s0" + str(i+1) + "-02_img.tif"
        
files.remove(None)
for i in range (0, n-2):
    if files[i] != None:
        print(files[i])
        shutil.copy(files[i], "data/humans/sad")