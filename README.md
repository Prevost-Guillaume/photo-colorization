# photo-colorization
Autoencoder versus U-Net for photo colorization

## Motivation :
I wanted to compare different models for colorization of images, and get an eyeful of some nice mountain landscapes. What better way to do this than to colorize mountain landscapes?


## Overview
This project reviews the different aspects below.

 - Dataset creation : scrape over 2500 mountain pics from google image using Selenium
 - Chose loss function : use mse for simplicity, as a good metric to evaluate distance between two pixels
 - Creation of the models architectures : Build U-Net and Autoencoder architectures using Keras
 - Model training : 10h on CPU for each model, on 2000 training photos
 - Evaluating predictions, comparing results between both models.


More coming soon !
