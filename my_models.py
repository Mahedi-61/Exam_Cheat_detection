from keras.models import Model
from keras.layers import (Input, Dense, LSTM, Lambda, TimeDistributed, GRU, 
                          BatchNormalization, Bidirectional, Activation)

from keras import regularizers
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import backend as K

import numpy as np

# project modules
from .. import config


nb_steps = config.nb_steps
nb_features = config.nb_features
nb_classes = config.nb_classes
rnn_model_path = config.rnn_model_path



### special layer
class CenterLossLayer(Layer):

    def __init__(self, alpha = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                            shape = (nb_classes, config.nb_cells), # (class, features) (8, 24)
                            initializer = 'uniform',
                            trainable = True)
 
        super().build(input_shape)

    def call(self, x, mask = None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis = 1, keepdims = True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)




### model
def temporal_network (x, labels):
    x = BatchNormalization(momentum = 0.92,
                    epsilon = 1e-5)(x)
    #
    # hidden GRU layers
    for i in range(config.nb_layers):
        x = Bidirectional(GRU(
            config.nb_cells,
            return_sequences = True,
            stateful = False),
            merge_mode = "sum")(x)
    
    # x --> (28, 80)
    main = Dense(nb_classes, 
            kernel_regularizer = regularizers.l2(0.01)) (x) 
    
    main = BatchNormalization(momentum = 0.92, epsilon = 1e-5) (main)
    main = Activation('softmax')(main)

    # making x --> (?, 80) and labels --> (?, 62)
    labels = Lambda(lambda x: K.max(x, axis = 1, keepdims = False))(labels)
    x = Lambda(lambda x: K.mean(x, axis = 1, keepdims = False))(x)

    side = CenterLossLayer(alpha = 0.5, name = 'centerlosslayer')([x, labels])
    return main, side



def get_temporal_model():

    ### compile
    main_input = Input((nb_steps, nb_features))  #(28, 20)
    aux_input = Input((nb_steps, nb_classes)) #(28, 8)

    final_output, side_output = temporal_network(main_input, aux_input)
    model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])

    # saving as json file in model directory
    open(rnn_model_path, 'w').write(model.to_json())

    print("model json saved !!")
    return model


if __name__ == "__main__":
    model = get_temporal_model()
    model.summary()