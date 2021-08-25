# photo-colorization
Autoencoder and ResNet versus U-Net for photo colorization

## Motivation :
I wanted to compare different models for colorization of images, and get an eyeful of some nice mountain landscapes. What better way to do this than to colorize mountain landscapes?


## Overview
This project reviews the different aspects below.

 - Dataset creation : scrape over 2500 mountain pics from google image using Selenium
 - Chose loss function : use mse for simplicity, as a good metric to evaluate distance between two pixels
 - Creation of the models architectures : Build U-Net and Autoencoder architectures using Keras
 - Model training : 10h on CPU for each model, on 2000 training photos
 - Evaluating predictions, comparing results between both models.

## Examples
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_1.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_2.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_3.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_4.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_5.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_6.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_7.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_8.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_9.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_10.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_11.png>
<img src=https://github.com/Prevost-Guillaume/photo-colorization/blob/main/images/Figure_12.png>

## Analysis

It is obvious to note that the Unet model obtains the best results. The colors are more vivid and the borders are fine.
The autoencoder and the ResNet are very similar. The ResNet just tends to put a bit of blue where it shouldn't.

Here are the loss and the accuracy for each model :

| model  | Loss  | Accuracy |
| ------------- | ------------- | ------------- |
| Autoencoder  | 0.00312  | 0.747  |
| ResNet  | 0.00335  | 0.719  |
| U-Net  | 0.00350  | 0.755  |



Accuracy seems to be a surprisingly good metric because, despite its highest loss, the U-Net model obtains the best results.

## Conclusion
Despite the ubiquity of autoencoders in the image colorization task in the literature, the U-Net model seems extremely promising, because it takes more risks. 
It should be verified that this tendency is also manifested in all types of colorisaton (not only in restricted areas like mountains).

An axis of improvement is to train the models on more data.

