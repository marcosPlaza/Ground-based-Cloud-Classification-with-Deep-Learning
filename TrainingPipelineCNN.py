from DataLoader import DataLoader
from LearningRateSchedulers import StepDecay
from datetime import datetime
import math
import random
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image, ImageEnhance
from tqdm import tqdm
import visualkeras

from tensorflow.keras.applications import ResNet152V2, VGG16, MobileNet, EfficientNetB0, MobileNetV2, InceptionResNetV2

def build_model_imagenet(input_shape, n_classes):
    pre_trained_model = MobileNetV2(input_shape=(256,256,3),
                         include_top = False,
                         weights='imagenet',
                         classes = n_classes,
                         classifier_activation='softmax')

    for layer in pre_trained_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(pre_trained_model)
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5)) 
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=n_classes, activation="softmax"))

    return model


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
    model.add(Dense(units=128, activation='relu')) # antes con 256

    # Output Layer 
    model.add(Dense(units=n_classes, activation='softmax'))

    return model

def build_pretrained_model(n_classes, pretrained_model):
    # Load the pretrained model
    base_model = load_model(pretrained_model)
    base_model.trainable =  True # False    # freeze layers

    model = Sequential()

    # paste pretrained layers 
    model.add(base_model.layers[0])
    model.add(base_model.layers[1])
    #model.add(base_model.layers[2])
    #model.add(base_model.layers[3])
    #model.add(base_model.layers[4])
    #model.add(base_model.layers[5])
    #model.add(base_model.layers[6])

    # Passing it to a dense layer
    model.add(Flatten())

    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Dense Layer
    model.add(Dense(units=128, activation='relu')) # antes con 256

    # Output Layer 
    model.add(Dense(units=n_classes, activation='softmax'))

    return model

def data_aug(src_X, height=256, width=256, channels=3, n_times=1):
    datagen = ImageDataGenerator(
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.75,1.35], # brightness
        zoom_range=[0.75,1.0]) # zoom

    # fit parameters from data
    datagen.fit(src_X)

    N = src_X.shape[0]
    X = np.zeros((N*n_times, height, width, channels))
    
    for n in range(n_times):
        for X_batch in datagen.flow(src_X, batch_size=N):
            for i in tqdm(range(0, N)):
                X[N*n+i,:,:,:] = X_batch[i]/255. # normalize images
            break

    return X


if __name__ == "__main__":
    tensorboard = sys.argv[1] == 'True' # True if you want to use tensorboard
    save_historic = sys.argv[2] == 'True' # True if you want to save the historic of the model
    scheduler = sys.argv[3] == 'True' # True if you want to use a scheduler
    pretrained = sys.argv[4] == 'True' # True if you want to use a pretrained model
    pretrained_model = sys.argv[5] # Path to the pretrained model
    dataset_path = sys.argv[6] # Path to the dataset
    model_path = sys.argv[7] # The model will be stored in this location
    history_path = sys.argv[8] # The historic will be stored in this location
    epochs = int(sys.argv[9]) # Number of epochs
    batch_size = int(sys.argv[10]) # Batch size
    initial_lr = float(sys.argv[11]) # Initial learning rate
    data_path = sys.argv[12] # Path to the data (dataset python object)
    data_from_file = sys.argv[13] == 'True' # True if you want to load the dataset from a file
    data_augmentation = sys.argv[14] == 'True' # True if you want to use data augmentation
    early_stopping = sys.argv[15] == 'True' # True if you want to use early stopping
    imagenet = sys.argv[16] == 'True' # True if you want to use imagenet pretrained model

    # print the arguments
    print("------------------- SESSION INFO --------------------")
    print("Tensorboard: ", 'ON' if tensorboard else 'OFF') 
    print("Model Path: ", model_path)
    print("Data Augmentation: ", 'ON' if data_augmentation else 'OFF')
    print("Early Stopping: ", 'ON' if early_stopping else 'OFF')
    print("Epochs: ", epochs)
    print("Batch Size: ", batch_size)
    print("Data from file: ", 'ON' if data_from_file else 'OFF') 
    if data_from_file: print("Data file: ", data_path)
    else: print("Dataset location: ", dataset_path)
    print("Save Historic: ", 'ON' if save_historic else 'OFF')
    if save_historic: print("History Path: ", history_path)
    print("Initial Learning Rate: ", initial_lr)
    print("Scheduler: ", 'ON' if scheduler else 'OFF')
    print("Pretrained: ", 'ON' if pretrained else 'OFF')
    if pretrained: print("Pretrained Model: ", pretrained_model)
    print("Pretrained imagenet: ", 'ON' if imagenet else 'OFF')
    print("------------------- STARTING TRAIN... ---------------")
    print("\tLOADING DATA...\n")

    image_size = 256
    n_channels = 3

    # regrouping the clouds
    # regroup = {'Sc':'Patterned Clouds', 'Ac':'Patterned Clouds', 'Ns':'Thick Dark Clouds', 'Ci':'Clear Sky', 'Cu':'Thin White Clouds', 'Cs':'Patterned Clouds', 'Ct':'Clear Sky', 'St':'Patterned Clouds', 'As':'Veil Clouds', 'Cc':'Patterned Clouds', 'Cb':'Thick White Clouds'}
    # regroup = {'Sc':'Patterned Clouds', 'Ac':'Patterned Clouds', 'Cu':'Thin White Clouds', 'As':'Veil Clouds', 'Cb':'Thick White Clouds'}
    # regroup = {'Sc':'Patterned Clouds', 'Ac':'Patterned Clouds', 'Ns':'Thick Dark Clouds', 'Ci':'Clear Sky', 'Cu':'Thin White Clouds', 'Cs':'Patterned Clouds', 'St':'Patterned Clouds', 'As':'Veil Clouds', 'Cc':'Patterned Clouds', 'Cb':'Thick White Clouds'}
    
    # Load the data
    train_data = DataLoader()
    
    if data_from_file: 
        train_data.load_from_file(data_path)

        print("\tDATA LOADED SUCCESSFULLY from {}".format(data_path))

    else: 
        train_data.load_data(dataset_path, image_size, n_channels, alt_classes=None)
    
        with open(data_path, 'wb') as datafile:
            pickle.dump(train_data, datafile, protocol=pickle.HIGHEST_PROTOCOL)

        print("\tDATA SAVED to {}\n".format(data_path))

    # split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(train_data.X, train_data.y, test_size=0.25, random_state=42)

    if data_augmentation:
        # data augmentation
        print("Before data augmentation... ", X_train.shape)

        X_tmp = []
        y_tmp = []
                    
        for idx, cl in enumerate(np.rot90(np.unique(y_train, axis=0))):
            X_n = X_train[np.where((y_train == cl).all(axis=1))[0]]
            X_n = data_aug(X_n, height=image_size, width=image_size, channels=3,  n_times=2)
            X_tmp.append(X_n)
            y_aug_tmp = np.zeros((X_n.shape[0], len(train_data.class_names)))
            for i in y_aug_tmp:
                i[idx] += 1
            y_tmp.append(y_aug_tmp)

        X_train = np.concatenate(X_tmp, axis=0)
        y_train = np.concatenate(y_tmp, axis=0)

        print("After data augmentation => ",X_train.shape)

    # Define the model
    if pretrained:
        model = build_pretrained_model(n_classes=train_data.class_names.shape[0], pretrained_model=pretrained_model)
    elif imagenet:
        model = build_model_imagenet(input_shape=(image_size, image_size, n_channels), n_classes=train_data.class_names.shape[0])
    else:
        model = build_model_from_scratch(input_shape=(image_size, image_size, n_channels), n_classes=train_data.class_names.shape[0])
    
    #compile model using accuracy to measure model performance
    model.compile(loss='categorical_crossentropy', optimizer= SGD(learning_rate= initial_lr, momentum=0.9), metrics=['accuracy'])

    print("\tMODEL SUMMARY")
    print(model.summary())
    visualkeras.layered_view(model, to_file='./Images/' + model_path[9:-3] + '_arc.png')

    steps_per_epoch = len(X_train)//batch_size
    validation_steps = len(X_val)//batch_size

    callbacks = []

    # the model is saved by default to .Models/<model_name> file
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    callbacks.append(checkpoint)
    
    if early_stopping: 
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
        callbacks.append(early) # uncomment to use early stopping

    # lr scheduler not used
    if scheduler: 
        lr_scheduler = LearningRateScheduler(StepDecay(initAlpha=initial_lr, factor=0.6, dropEvery=20), verbose=1)
        callbacks.append(lr_scheduler)

    # tensorboard
    if tensorboard:
        log_dir = "./Logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tb)
        
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
        plt.savefig('./Images/' + fname)

        print("\tSaved plot to: ", fname)

    
