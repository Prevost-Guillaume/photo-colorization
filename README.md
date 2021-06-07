# photo-colorization
Autoencoder versus U-Net for photo colorization

Motivation :
I wanted to compare different models for colorization of images, and get an eyeful of some nice mountain landscapes. What better way to do this than to colorize mountain landscapes?

(ConvNet, Autoencoder, DenseNet, UNet)


 - Dataset creation using Selenium (1000 mountain pics)
 - Chose loss function (mse for simplicity, good metric to evaluate distance between two pixels)
 - Creation of the models architectures (U-Net, Autoencoder) using Keras
 - Model training (10h on CPU for each model, on 700 training photos)
 - Evaluating predictions


More coming soon !
