from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras import backend

import config


# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = backend.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


def atari_model(n_actions):
    # With the functional API we need to define the inputs.
    frames_input = Input(config.atari_shape, name='frames')
    actions_input = Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = Lambda(lambda x: x / 255.0)(frames_input)

    # layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity.
    conv_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(normalized)

    # second layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(conv_1)

    conv_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(conv_2)

    # Flattening the second convolutional layer.
    conv_flattened = Flatten()(conv_3)

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = Dense(512, activation='relu')(conv_flattened)

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(n_actions)(hidden)

    # Finally, we multiply the output by the mask!
    filtered_output = multiply([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = RMSprop(lr=config.learning_rate, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)

    return model


def fit_batch(model, target_model, start_states, actions, rewards, next_states, is_terminals):
    """
    Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminals: numpy boolean array of whether the resulting state is terminal

    """

    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_q_values = target_model.predict([next_states, np.ones((config.batch_size, config.num_actions))])

    # The Q values of the terminal states is 0 by definition, so override them
    next_q_values[is_terminals] = 0

    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + config.gamma * np.max(next_q_values, axis=1)

    # Fit the keras model.
    # Note how we are passing the actions as the mask and multiplying the targets by the actions.
    actions_target = np.eye(config.num_actions)[np.array(actions).reshape(-1)]
    targets = actions_target * q_values[:, None]
    model.fit(x=[start_states, actions_target], y=targets, batch_size=config.batch_size, epochs=1, verbose=0)


def choose_best_action(model, state):
    state_reshape = np.reshape(state, (1, config.atari_shape[0], config.atari_shape[1], config.atari_shape[2]))
    q_value = model.predict([state_reshape, np.ones((1, config.num_actions))], batch_size=1)
    return np.argmax(q_value[0])
