# -*- coding: utf-8 -*-

from keras.layers import Convolution2D as Conv2D, Input, Dropout, Flatten, Dense, MaxPooling2D
#from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential

def get_model():
    model = Sequential()
    ### SOLUTION: Layer 1: Convolutional follwed by max pooling and dropout.
    model.add(Conv2D(2, 3, 3, subsample=(2, 2), input_shape=(16, 32, 1), border_mode='valid', activation='relu'))
    #### using 2, 4 as k is because the ratio of the size is 1:2.
    model.add(MaxPooling2D((2, 4), strides=(4, 4), border_mode='same'))
    #### although the size of tiny model is samll, dropout is still used to prevent overfitting.
    model.add(Dropout(0.25))

    ### SOLUTION: Layer 2: Flatten.
    model.add(Flatten())
    
    ### SOLUTION: Layer 3: Fully connected with a single output since this is a regression problem..
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


if __name__ == '__main__':
    import numpy as np
    from os import path
    import argparse

    parser = argparse.ArgumentParser(description='Training tiny model.')
    parser.add_argument(
        '--batch', type=int, default=128, help='batch size.')
    parser.add_argument(
        '--epoch', type=int, default=50, help='number of epochs.')
    args = parser.parse_args()

    # load training data with file name for tiny model.
    x = np.load('x.data.tiny.npy')
    y = np.load('y.data.tiny.npy')

    # get model
    model = get_model()
    # show model architecture
    model.summary()

    # export model architecture
    model_json = model.to_json()
    with open("model.tiny.json", "w") as f:
        f.write(model_json)
    #checkpoint = ModelCheckpoint("model.tiny.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    #early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4, verbose=1, mode='min')
    #history = model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, verbose=1, callbacks=[checkpoint, early_stop], validation_split=0.2, shuffle=True)
    history = model.fit(
        x, 
        y, 
        batch_size=args.batch, 
        nb_epoch=args.epoch, 
        verbose=1, 
        validation_split=0.2, 
        shuffle=True)
    model.save_weights('model.tiny.h5')
    