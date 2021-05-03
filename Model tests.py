import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import warnings
warnings.simplefilter("ignore")

import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import pickle
import time

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, SpatialDropout2D
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization, InputLayer, LocallyConnected2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape

from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


from sklearn.utils import shuffle
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from sklearn.model_selection import train_test_split

import random







#############################################################################################################################################
#                                                                                                                                           #
#                                                             DATA FUNCTIONS                                                                #
#                                                                                                                                           #
#############################################################################################################################################

def getPictures(picNames, n='all', size=(28,28), color=False):
    X = []
    if n=='all' or n==-1:
        n = len(picNames)
    for j in range(n):
        i = picNames[j]
        if color:
           img = cv2.imread(i)
        else:
            img = cv2.imread(i,0)
        try:
            img = cv2.resize(img, size)
            X.append(img)
        except:
            pass
            #print(i)
        
    return X


def f(pic):
    """Fonction à appliquer à chaque image"""
    pic_lab = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)
    pic_ab = pic_lab[:,:,1:]
    return pic_ab/255
            
        
def extractImages(fileName, size=(128,128), split=False):
    """Génère X_train, y_train, X_test, y_test, input_shape, output_shape, dico_labels"""
    if split==False:
        split = 1
        
    input_shape=(size[0],size[1],1)
    output_shape = (size[0],size[1],2)

    pics = []
    labels = []
    
    pics_names = [fileName+'/'+f for f in listdir(fileName) if f[-4:] == '.jpg' or f[-4:] == '.png']
    pics_color = getPictures(pics_names, n='all', size=(size[0],size[1]), color=True)
    pics_gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in pics_color]

        
    pics_ab = [f(np.array(im)) for im in pics_color]
    
    z = list(zip(pics_gray, pics_ab))
    random.shuffle(z)
    pics_gray, pics_ab = zip(*z)

    X_train, y_train = np.array(pics_gray), np.array(pics_ab)

    
    X_test, y_test = [],[]
    if split != False:
        split = int(split*len(y_train))
        X_train, y_train,X_test, y_test = X_train[:split], y_train[:split], X_train[split:], y_train[split:]

    X_train = X_train.reshape(X_train.shape[0], size[0], size[1], 1)
    X_test = X_test.reshape(X_test.shape[0], size[0], size[1], 1)
    y_train = y_train.reshape(y_train.shape[0], size[0], size[1], 2)
    y_test = y_test.reshape(y_test.shape[0], size[0], size[1], 2)
    
    return X_train, y_train, X_test, y_test, input_shape, output_shape



def getOriginalFromAB(pic_ab, pic_grey):
    NewImg = np.zeros((pic_ab.shape[0],pic_ab.shape[1],3), dtype=np.uint8)
    for line in range(pic_ab.shape[0]):
        for col in range(pic_ab.shape[1]):
            NewImg[line, col] = (pic_grey[line, col],255*pic_ab[line, col][0],255*pic_ab[line, col][1])
    return cv2.cvtColor(NewImg, cv2.COLOR_LAB2BGR)
    





#############################################################################################################################################
#                                                                                                                                           #
#                                                                   MAIN                                                                    #
#                                                                                                                                           #
#############################################################################################################################################

data_file = "C:/Users/prevo/OneDrive/Documents/Programmes Python/Keeras/Complete projects/EN COURS - Pics colorization/testset"
size=(128,128)
X, y, _, _, input_shape, output_shape = extractImages(data_file, size=size, split=1)


print('Load model from disk')
model1 = load_model("model_Autoencoder_200_epochs.h5")
model2 = load_model("model_conv_100_epochs.h5")
model3 = load_model("model_Reverse_AE_200_epochs.h5")
model4 = load_model("model_UNet_blocs_200_epochs.h5")

#print(model4.summary())




for i in range(len(X)):
    pic_gray = X[i]
    x = np.array([pic_gray])

    pic_ab1 = model1.predict(x)
    pic_ab2 = model2.predict(x)
    pic_ab3 = model3.predict(x)
    pic_ab4 = model4.predict(x)
    new_pic1 = getOriginalFromAB(pic_ab1[0], pic_gray)
    new_pic2 = getOriginalFromAB(pic_ab2[0], pic_gray)
    new_pic3 = getOriginalFromAB(pic_ab3[0], pic_gray)
    new_pic4 = getOriginalFromAB(pic_ab4[0], pic_gray)

    pic_ab_true = y[i]
    new_pic_true = getOriginalFromAB(pic_ab_true, pic_gray)

    new_pic1 = cv2.cvtColor(new_pic1, cv2.COLOR_BGR2RGB)
    new_pic2 = cv2.cvtColor(new_pic2, cv2.COLOR_BGR2RGB)
    new_pic3 = cv2.cvtColor(new_pic3, cv2.COLOR_BGR2RGB)
    new_pic4 = cv2.cvtColor(new_pic4, cv2.COLOR_BGR2RGB)
    new_pic_true = cv2.cvtColor(new_pic_true, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1,5, figsize = (12,4))
    x = np.arange(1,11)
    axes[0].imshow(new_pic1)
    axes[0].set_title('model 1')
    axes[1].imshow(new_pic2)
    axes[1].set_title('model 2')
    axes[2].imshow(new_pic3)
    axes[2].set_title('model 3')
    axes[3].imshow(new_pic4)
    axes[3].set_title('model 4')
    axes[4].imshow(new_pic_true)
    axes[4].set_title('original')
    fig.tight_layout()
    plt.show()


##    cv2.imshow('1',new_pic1)
##    cv2.imshow('2',new_pic2)
##    cv2.imshow('cible',new_pic_true)
##    #cv2.imshow('original',pic_gray)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()









s