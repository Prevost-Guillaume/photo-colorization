import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import warnings
warnings.simplefilter("ignore")
import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import pickle
import random

#Imports Packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


#############################################################################################################################################
#                                                                                                                                           #
#                                                               WEB SCRAPPING                                                               #
#                                                                                                                                           #
#############################################################################################################################################

def downloadImages(keyword, n=10, file='web_scrapping'):
    #Opens up web driver and goes to Google Images
    driver = webdriver.Chrome('C:/chromedriver.exe')
    driver.get('https://www.google.ca/imghp?hl=en&tab=ri&authuser=0&ogbl')

    box = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
    box.send_keys(keyword) 
    box.send_keys(Keys.ENTER)

    # Continuera à faire défiler la page Web jusqu'à ce qu'elle ne puisse plus faire défiler
    last_height = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        try:
            driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
            time.sleep(2)
        except:
            pass
        if new_height == last_height:
            break
        last_height = new_height

    for i in range(1, n+1):
        try:
            driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot(keyword+' ('+str(i)+').png')
        except:
            print('error')
            pass
        
    return


def resizePics(size, file='web_scrapping'):
    
    img = cv2.imread(i)
    img = cv2.resize(img, size)

#############################################################################################################################################
#                                                                                                                                           #
#                                                                MAIN                                                                       #
#                                                                                                                                           #
#############################################################################################################################################
keyword='alpes été' # hiking, alps, summer nepal, etc.
downloadImages(keyword, n=100, file='web_scrapping')




