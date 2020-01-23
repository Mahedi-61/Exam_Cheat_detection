"""
Author: A Cup of Tea
"""

import keras
import numpy as np 
import os

# project modules
from .. import config
from . import my_model, make_dataset_3dcd


# loading data
X_train, y_train = make_dataset_3dcd.get_train_data()

print("train data shape: ", X_train.shape)
print("train data label: ", y_train.shape)

# for i, ar in enumerate(y_train): print("label: ", i, " value: ", ar)


# laoding model
model = my_model.get_model()

# compile
model.compile(keras.optimizers.Adam(config.lr), 
            keras.losses.categorical_crossentropy,
            metrics=['accuracy'])


# checkpoins
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()


# for training model
model.fit(X_train, y_train, 
        batch_size = config.batch_size, 
        epochs =config.nb_epochs,  
        verbose = 2, 
        shuffle = True, 
        callbacks = [early_stopping, model_cp], 
        validation_split = 0.15)

