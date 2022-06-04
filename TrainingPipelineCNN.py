from DataLoader import DataLoader
from LearningRateSchedulers import StepDecay
import math
import random
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image, ImageEnhance
from tqdm import tqdm
from tensorboard import program


# Define the model
def build_model_from_scratch(input_shape, n_classes):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # Passing it to a dense layer
    model.add(Flatten())

    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Dense Layer
    model.add(Dense(units=256, activation='relu'))

    # Output Layer 
    model.add(Dense(units=n_classes, activation='softmax'))

    return model

def build_pretrained_model(n_classes, pretrained_model):
    # Load the pretrained model
    base_model = load_model(pretrained_model)
    base_model.trainable = False    # freeze layers

    model = Sequential()

    # paste pretrained layers 
    model.add(base_model.layers[0])
    model.add(base_model.layers[1])
    model.add(base_model.layers[2])
    model.add(base_model.layers[3])
    model.add(base_model.layers[4])
    model.add(base_model.layers[5])
    model.add(base_model.layers[6])

    # Passing it to a dense layer
    model.add(Flatten())

    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Dense Layer
    model.add(Dense(units=256, activation='relu'))

    # Output Layer 
    model.add(Dense(units=n_classes, activation='softmax'))

    return model


if __name__ == "__main__":
    tensorboard = sys.argv[1] == 'True'
    save_historic = sys.argv[2] == 'True'
    scheduler = sys.argv[3] == 'True'
    pretrained = sys.argv[4] == 'True'
    pretrained_model = sys.argv[5]
    abs_path = sys.argv[6]
    model_path = sys.argv[7]
    history_path = sys.argv[8]
    epochs = int(sys.argv[9])
    batch_size = int(sys.argv[10])
    initial_lr = float(sys.argv[11])
    save_path = sys.argv[12]
    data_from_file = sys.argv[13] == 'True'
    data_augmentation = sys.argv[14] == 'True'
    early_stopping = sys.argv[15] == 'True'

    # print the arguments
    print("------------------- SESSION INFO --------------------")
    print("Tensorboard: ", 'ON' if tensorboard else 'OFF') # 1
    print("Save Historic: ", 'ON' if save_historic else 'OFF') # 2
    print("Scheduler: ", 'ON' if scheduler else 'OFF') # 3
    print("Pretrained: ", 'ON' if pretrained else 'OFF') # 4
    print("Data Augmentation: ", 'ON' if data_augmentation else 'OFF') # 6
    print("Early Stopping: ", 'ON' if early_stopping else 'OFF') # 7
    print("Data from file: ", 'ON' if data_from_file else 'OFF') # 8
    print("Initial Learning Rate: ", initial_lr) # 5
    print("Pretrained Model: ", pretrained_model) # 9
    print("Data location: ", abs_path) # 10
    print("Model Path: ", model_path) # 11
    print("History Path: ", history_path) # 12
    print("Epochs: ", epochs) # 13
    print("Batch Size: ", batch_size) # 14
    print("------------------- STARTING TRAIN... ---------------")
    print("\tLOADING DATA...\n")

    image_size = 227
    n_channels = 3

    # Load the data
    train_data = DataLoader()
    
    if data_from_file: 
        train_data.load_from_file(save_path)

        print("\tDATA LOADED SUCCESSFULLY from {}".format(save_path))

    else: 
        train_data.load_data(abs_path, image_size, n_channels)
    
        with open(save_path, 'wb') as datafile:
            pickle.dump(train_data, datafile, protocol=pickle.HIGHEST_PROTOCOL)

        print("\tDATA SAVED to {}\n".format(save_path))

    # split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(train_data.X, train_data.y, test_size=0.3, random_state=42)

    # Define the model
    if pretrained:
        model = build_pretrained_model(n_classes=train_data.class_names.shape[0], pretrained_model=pretrained_model)
    else:
        model = build_model_from_scratch(input_shape=(image_size,image_size,n_channels), n_classes=train_data.class_names.shape[0])
    
    #compile model using accuracy to measure model performance
    model.compile(loss='categorical_crossentropy', optimizer= SGD(learning_rate= initial_lr, momentum=0.9), metrics=['accuracy'])

    steps_per_epoch = len(X_train)//batch_size
    validation_steps = len(X_val)//batch_size

    callbacks = []

    # the model is saved by default to .Models/<model_name> file
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    callbacks.append(checkpoint)
    
    if early_stopping: callbacks.append(early) # uncomment to use early stopping

    # lr scheduler not used
    if scheduler: 
        lr_scheduler = LearningRateScheduler(StepDecay(initAlpha=initial_lr, factor=0.6, dropEvery=20), verbose=1)
        callbacks.append(lr_scheduler)

    # tensorboard
    if tensorboard:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', "./Logs/"])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")

    # fit the model
    if data_augmentation:
        print("Before ", X_train.shape)

        # create data generator
        datagen = ImageDataGenerator(
            horizontal_flip=True, # horizontal flip
            brightness_range=[0.85,1.25], # brightness
            zoom_range=[0.85,1.0])

        # fit parameters from data
        datagen.fit(X_train)

        N = X_train.shape[0]

        X2 = np.zeros((N, image_size, image_size, n_channels))
        y2 = np.zeros((N, train_data.class_names.shape[0]))

        # Configure batch size and retrieve one batch of images
        print("\tAUGMENTING DATA x2...\n")
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=N):
            for i in tqdm(range(0, N, 1)):
                X2[i,:,:,:] = X_batch[i].reshape(image_size, image_size, n_channels)/255.
                y2[i] = y_batch[i]
            # De esta manera paramos de generar imagenes aleatoriamente
            break
            
        X_train = np.concatenate((X_train,X2))
        y_train = np.concatenate((y_train,y2))

        X3 = np.zeros((N, image_size, image_size, n_channels))
        y3 = np.zeros((N, train_data.class_names.shape[0]))

        # Configure batch size and retrieve one batch of images
        print("\tAUGMENTING DATA x2 (2)...\n")
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=N):
            for i in tqdm(range(0, N, 1)):
                X3[i,:,:,:] = X_batch[i].reshape(image_size, image_size, n_channels)/255.
                y3[i] = y_batch[i]
            # De esta manera paramos de generar imagenes aleatoriamente
            break
            
        X_train = np.concatenate((X_train,X3))
        y_train = np.concatenate((y_train,y3))

        print("After ",X_train.shape)
        
        print("\tTRAINING...\n")
        
        history = model.fit(
            X_train, 
            y_train, 
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=(X_val, y_val),
            callbacks=callbacks)
    else:
        print("\tTRAINING...\n")

        history = model.fit(
            X_train, 
            y_train, 
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=(X_val, y_val),
            callbacks=callbacks)

    if save_historic:
        # save the training history
        with open(history_path, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # show the accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])

        # save plot by default
        splits = model_path.split('/')
        splits = splits[-1].split('.')
        fname = splits[0] + '_accuracy_loss.png'
        plt.savefig(fname)

    
