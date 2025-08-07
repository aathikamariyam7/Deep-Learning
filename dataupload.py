import os
import tkinter as tk
import time
from random import shuffle

import PIL.Image as pimg
import PIL.ImageTk as pimgtk

from tkinter import *
from tkinter import messagebox, filedialog
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from keras.preprocessing.image import ImageDataGenerator
import tk as tk
import tkinter as tk

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
batch_size = 32


from PIL import Image, ImageTk
from glob import glob

from PIL.ImageFile import ImageFile
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model


class ViewData:
    def __init__(self):
        def dataupload():
            s1 = "A:\Tooth_Detection\Dataset\Train"
            messagebox.showinfo("Success", s1)

        def viewdata():
            workingDir = "A:\Tooth_Detection\Dataset"
            PATH = os.path.sep.join([workingDir, "Train"])
            train_dir = os.path.join("A:\Tooth_Detection\Dataset", "Train")

            # Getting the path to the validation directory
            validation_dir = os.path.join("A:\Tooth_Detection\Dataset", "Train")

            train_dir1 = os.path.join(train_dir, "Calculus")
            train_dir2 = os.path.join(train_dir, "caries")
            train_dir3 = os.path.join(train_dir, "hypodontia")


            # Getting the path to the directory for the parasitized validation cell images and
            # the path to the directory for the uninfected validation cell images
            val_dir1 = os.path.join(validation_dir, "Calculus")
            val_dir2 = os.path.join(validation_dir, "caries")
            val_dir3 = os.path.join(validation_dir, "hypodontia")


            # Getting the number of images present in the parasitized training directory and the
            # number of images present in the uninfected training directory
            train_images1 = len(os.listdir(train_dir1))
            train_images2 = len(os.listdir(train_dir2))
            train_images3 = len(os.listdir(train_dir3))



            # Getting the number of images present in the parasitized validation directory and the
            # number of images present in the uninfected validation directory
            images_val1 = len(os.listdir(val_dir1))
            images_val2 = len(os.listdir(val_dir2))
            images_val3 = len(os.listdir(val_dir3))



            # Getting the sum of both the training images and validation images
            total_train =  train_images1 + train_images2+train_images3
            total_val = images_val1 + images_val2+images_val3

            print("Total Train Images: {}".format(total_train));
            #print("Total Validation: {}".format(total_val));
            s1 = "Total Train Image "+format(total_train)
            messagebox.showinfo("Success", s1)
           # messagebox.showinfo("Extraction Sucessfully")

        def viewdata1():
            workingDir = "A:\Tooth_Detection\Dataset"
            PATH = os.path.sep.join([workingDir, "Train"])

            validation_dir = os.path.join("A:\Tooth_Detection\Dataset", "Train")
            train_dir = os.path.join("A:\Tooth_Detection\Dataset", "Train")
            batch_size = 2000
            epochs = 20
            IMG_HEIGHT = 98
            IMG_WIDTH = 98

            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                shuffle=True
                                                                )

            validation_datagen = ImageDataGenerator(rescale=1. / 255)

            validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                          batch_size=batch_size,
                                                                          class_mode='binary',
                                                                          shuffle=True
                                                                          )
            s1 = "Total Class Extracted Sucessfully"
            messagebox.showinfo(" Success", s1)
        def build():
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            train_path = 'Dataset/Train'
            valid_path = 'Dataset/Test'

            # load model without output layer

            IMAGE_SIZE = [300, 300]
            # add preprocessing layer to the front of VGG
            vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

            # don't train existing weights
            for layer in vgg.layers:
                layer.trainable = False

            # useful for getting number of classes
            folders = glob('Dataset/Train/*')

            x = Flatten()(vgg.output)
            # x = Dense(1000, activation='relu')(x)
            prediction = Dense(len(folders), activation='softmax')(x)

            # create a model object
            model = Model(inputs=vgg.input, outputs=prediction)

            # view the structure of the model
            model.summary()

            # tell the model what cost and optimization method to use
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # Use the Image Data Generator to import the images from the

            from keras.preprocessing.image import ImageDataGenerator
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            val_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                             target_size=(300, 300),
                                                             batch_size=32,
                                                             class_mode='categorical')

            val_set = val_datagen.flow_from_directory('Dataset/Test',
                                                      target_size=(300, 300),
                                                      batch_size=32,
                                                      class_mode='categorical')

            # fit the model
            history = model.fit_generator(training_set,
                                          validation_data=val_set,
                                          epochs=5,
                                          steps_per_epoch=len(training_set),
                                          validation_steps=len(val_set))

            model.save('model_tooth.h5')
            s1="Model Build Sucessfully"
            messagebox.showinfo(" Success",s1)


           # print(pd.DataFrame(history.history))

           # pd.DataFrame(history.history).plot()





        win = Tk()
        win.title("Dental Caries Detection")
        win.maxsize(width=900, height=800)
        win.minsize(width=900, height=800)
        win.configure(bg='#99ddff')

        image1 = Image.open("2.jpg")
        img = image1.resize((900, 800))

        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(win, image=test)
        label1.image = test

        # Position image
        label1.place(x=1, y=1)

        # image1 = Image.open("3.png")
        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(win, image=test)
        label1.image = test

        Label(win, text='Detecting Tooth Caries with CNN Precision', bg="#34bfbb", font='verdana 15 bold') \
            .place(x=170, y=120)
        btnbrowse = Button(win, text="Dataset source", font=' Verdana 10 bold', command=lambda: dataupload())
        btnbrowse.place(x=70, y=200)

        btncamera = Button(win, text="Trained Image Extraction", font='Verdana 10 bold', command=lambda: viewdata())
        btncamera.place(x=220, y=200)

        btnsend = Button(win, text="Total Class Extraction", font='Verdana 10 bold', command=lambda: viewdata1())
        btnsend.place(x=450, y=200)

        btnsend = Button(win, text="Build CNN Model", font='Verdana 10 bold', command=lambda: build())
        btnsend.place(x=650, y=200)

        win.mainloop()
