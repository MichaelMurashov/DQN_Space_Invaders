import numpy as np


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def preprocess_state(state):
    result = []
    for i in state:
        result.append(preprocess(i))
    return result


def transform_reward(reward):
    return np.sign(reward)
