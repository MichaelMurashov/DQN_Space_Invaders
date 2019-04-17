from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop

import config


def atari_model(n_actions):
    # With the functional API we need to define the inputs.
    frames_input = Input(config.atari_shape, name='frames')
    actions_input = Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = Lambda(lambda x: x / 255.0)(frames_input)

    # layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity.
    conv_1 = Conv2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)

    # second layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = Conv2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)

    # Flattening the second convolutional layer.
    conv_flattened = Flatten()(conv_2)

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = Dense(256, activation='relu')(conv_flattened)

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(n_actions)(hidden)

    # Finally, we multiply the output by the mask!
    # filtered_output = merge([output, actions_input], mode='mul')
    filtered_output = Multiply([output, actions_input])

    model = Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = RMSprop(lr=config.learning_rate, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model
