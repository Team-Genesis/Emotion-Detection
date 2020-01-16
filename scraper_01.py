#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:49:50 2020

@author: ishant
"""

import urllib.request
import random
from selenium import webdriver
import os
from datetime import datetime

start = datetime.now()


def scraper(url):
    global count
    driver = webdriver.Firefox(executable_path = "/home/ishant/ishant_linux/facial-emotion-recognition/geckodriver")
    driver.get(url)
    results = driver.find_elements_by_class_name("z_g_a")
    

    for result in results:
        if result.get_attribute('src') == None:
            print("the error is here")
            continue
        img_name = "neutral" + str(random.randrange(10,100000)) + ".jpg"
        fullfilename = os.path.join("data/scraper/neutral", img_name)
        urllib.request.urlretrieve(result.get_attribute('src'), fullfilename)
        # print(fullfilename + " Successfully Saved!")
        count =+ 1
    driver.quit()
    print("Scraper function successfully executed")
    
urlpage = [None] * 5
for i in range (0, 5): 
    if i==0:
        urlpage[i] = "https://www.shutterstock.com/search/neutral+facial+Expression"
    else:
        urlpage[i] = "https://www.shutterstock.com/search/neutral+facial+Expression?page=" + str(i+1)


print(urlpage)


count = 0
# print(urlpage)
for x in urlpage:
    scraper(x)
    print("Scraper function successfully called and executed")

end = datetime.now()
total_time = end - start
print("Total number of files saved :: " + str(count) + " in Time :: " + str(total_time) )
    




    