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
    



def evaluate(model, X_train, y_train, X_test, y_test):
    score1 = model.evaluate(X_train, y_train, verbose=0)
    score2 = model.evaluate(X_test, y_test, verbose=0)
    print("Score train : ",score1)
    print("Score test : ",score2)
    
    for i in range(len(X_test)):
        pic_gray = X_test[i]
        x = np.array([pic_gray])

        pic_ab = model.predict(x)
        new_pic = getOriginalFromAB(pic_ab[0], pic_gray)
        
        pic_ab_true = y_test[i]
        new_pic_true = getOriginalFromAB(pic_ab_true, pic_gray)
        
        cv2.imshow('prédite',new_pic)
        cv2.imshow('cible',new_pic_true)
        cv2.imshow('original',pic_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


#############################################################################################################################################
#                                                                                                                                           #
#                                                            MODEL FUNCTIONS                                                                #
#                                                                                                                                           #
#############################################################################################################################################
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    g = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(layer_in)
    g = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(g)
    if batchnorm:
        g = BatchNormalization()(g)
    g = MaxPooling2D(pool_size=(2, 2))(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    g = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(layer_in)
    if dropout:
        g = Dropout(0.2)(g)
    g = concatenate([g, skip_in], axis=3)
    g = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(g)
    g = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(g)
    return g

def getUnet(image_shape):
    print('nice UNet')

    #  Input
    in_image = Input(shape=(None, None, 1))

    # encoder
    e1 = define_encoder_block(in_image, 16, batchnorm=False)
    e2 = define_encoder_block(e1, 32, batchnorm=True)
    e3 = define_encoder_block(e2, 64, batchnorm=True)
    e4 = define_encoder_block(e3, 128, batchnorm=True)

    # Center
    b = Conv2D(64, (4,4), strides=(2,2), activation='relu', padding='same')(e4)

    # decoder
    d4 = decoder_block(b, e4, 128, dropout=True)
    d5 = decoder_block(d4, e3, 64, dropout=True)
    d6 = decoder_block(d5, e2, 32, dropout=True)
    d7 = decoder_block(d6, e1, 16, dropout=True)
    
    # output
    out_image = Conv2DTranspose(2, (4,4), strides=(2,2), activation='tanh', padding='same')(d7)
    
    # define model
    model = Model(in_image, out_image)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    
    return model

 


def getAutoencoder(input_shape):
    print('AE strides')
    model = Sequential()

    # Input
    model.add(InputLayer(input_shape=input_shape))

    # Encoder
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))

    # Decoder
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

    # Finish model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    return model


def getConv(input_shape):
    print('Straight Conv')
    model = Sequential()

    # Input
    model.add(InputLayer(input_shape=(None, None, 1)))

    # Conv Layers
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

    # Finish model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return model



def getReverseAutoencoder(input_shape):
    print('reverse AE')
    model = Sequential()
    
    # Input
    model.add(InputLayer(input_shape=(None, None, 1)))

    # Decoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

    # Encoder
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

    # Finish model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    return model







#############################################################################################################################################
#                                                                                                                                           #
#                                                                   MAIN                                                                    #
#                                                                                                                                           #
#############################################################################################################################################

createModel = True
modelName = "model_conv_100_epochs.h5"

data_file = "C:/Users/prevo/OneDrive/Documents/Programmes Python/Keeras/Complete projects/EN COURS - Pics colorization/web scrapping"
size=(128,128)
X_train, y_train, X_test, y_test, input_shape, output_shape = extractImages(data_file, size=size, split=0.9)


#
# TO DO : DATA AUGMANTATION
#

for i in range(0):
    pic_gray = X_train[i]
    pic_ab = y_train[i]
    new_pic = getOriginalFromAB(pic_ab, pic_gray)
    cv2.imshow('new_pic_train',new_pic)
    cv2.imshow('pic_gray_train',pic_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if createModel == True:
    # Create model
    model = getConv(input_shape)
    print(model.summary())

    # Fit model
    print('Train on ',len(X_train),' samples')
    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2)

    # save model and architecture to single file
    model.save(modelName)
    print("Saved model to disk")

else:
    model = load_model(modelName)
    print('Load model from disk')


evaluate(model, X_train, y_train, X_test, y_test)



