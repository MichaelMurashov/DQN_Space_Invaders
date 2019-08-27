min_frames_for_train = 100
buffer_size = 50000
num_iterations = 100000
atari_shape = (4, 105, 80)

num_frames = 4
num_actions = 6

batch_size = 32
learning_rate = 0.00025

initial_epsilon = 1.0
final_epsilon = 0.05
final_iteration_num = 10000

gamma = 0.99  # decay rate of past observations
