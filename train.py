""" train a rnn network using pose sequence of Exam cheat dataset"""

# python packages
import numpy as np
import os
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import LearningRateScheduler


# project modules
from . import my_models
from . import model_utils
from . import make_dataset_3dcd
from .. import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def lr_scheduler(epoch):
    if (epoch == 50):
        K.set_value(model.optimizer.lr, config.lr_1)

    elif (epoch == 100):
        K.set_value(model.optimizer.lr, config.lr_2)

    elif (epoch == 170):
        K.set_value(model.optimizer.lr, config.lr_3)
        
    print("learning rate: ", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


### custom loss
def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis = 0)



X_train, X_valid, y_train, y_valid = make_dataset_3dcd.get_train_data()
change_lr = LearningRateScheduler(lr_scheduler)


print(y_train.shape)
print(y_valid.shape)

# constructing model
model = my_models.get_temporal_model()

# train model once again
#model = model_utils.read_rnn_model(angle)


### run model
lambda_centerloss = 0.008

optimizer = Adam(lr = config.learning_rate)
model.compile(optimizer = optimizer,
                loss=[losses.categorical_crossentropy, zero_loss],
                loss_weights=[1, lambda_centerloss],
                metrics=['accuracy'])


# training and evaluating model
model_cp = model_utils.save_rnn_model_checkpoint()
early_stop = model_utils.set_early_stopping()


# fit
#y_train_value = np.argmax(y_train, axis = 2)
#y_valid_value = np.argmax(y_valid, axis = 2)

random_y_train = np.random.rand(X_train.shape[0], 1)
random_y_valid = np.random.rand(X_valid.shape[0], 1)


model.fit([X_train, y_train], [y_train, random_y_train], 
            batch_size = config.batch_size,
            shuffle = True,
            epochs = config.nb_epochs,
            callbacks = [change_lr, model_cp],
            verbose = 2,
            validation_data=([X_valid, y_valid], [y_valid, random_y_valid]))