import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import cube as cube_lib

HIDDEN_LAYERS_WIDTH = [1024, 512, 256]


def create_model() -> tf.keras.Model:
    '''Returns a compiled model.'''
    inputs = tf.keras.Input(shape=(20, 24))

    activation = tf.keras.layers.Flatten()(inputs)

    for layer_width in HIDDEN_LAYERS_WIDTH:
        layer = tf.keras.layers.Dense(layer_width, activation='relu')
        activation = layer(activation)

    output = tf.keras.layers.Dense(1, activation='linear')(activation)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError())

    return model


def random_rotation() -> Tuple[cube_lib.Face, bool]:
    '''Generates one random rotation of the cube.'''
    face = cube_lib.Face(random.randrange(0, cube_lib.NUM_FACES))
    clockwise = random.random() > 0.5
    return (face, clockwise)


def simulate_trajectory(trajectory_length: int) -> np.array:
    '''Simulates a trajectory that finishes in a solved state.

    Returns:
        an array of shape (trajectory_length, 20, 24).
    '''
    rotations = [random_rotation() for _ in range(trajectory_length)]

    states = []
    cube = cube_lib.Cube()
    for face, clockwise in rotations:
        cube.rotate_face(face, clockwise)
        states.append(cube.as_numpy_array())

    states = list(reversed(states))
    return np.array(states)


BATCH_SIZE = 32
NUM_EPOCHS = 1000000


def train_model(model: tf.keras.Model) -> None:
    '''Takes a compiled model and trains it.'''
    for epoch in range(NUM_EPOCHS):
        states = simulate_trajectory(BATCH_SIZE)
        labels = np.arange(BATCH_SIZE, 0, -1)
        verbose = 2 if epoch % 1000 == 0 else 0
        model.fit(x=states, y=labels, verbose=verbose)


def main():
    model = create_model()
    train_model(model)


if __name__ == "__main__":
    main()
