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
    is_clockwise = random.random() > 0.5
    return cube_lib.Rotation(face, is_clockwise)


TRAJECTORY_LENGTH = 32


def generate_training_example() -> Tuple[np.ndarray, int]:
    '''Generates training examples.'''
    while True:
        cube = cube_lib.Cube()
        for i in range(TRAJECTORY_LENGTH):
            cube.rotate_face(random_rotation())
            yield (cube.as_numpy_array(), i)


BATCH_SIZE = 32
MODEL_PATH = ''


def train_model(model: tf.keras.Model) -> None:
    '''Takes a compiled model and trains it.'''
    examples = tf.data.Dataset.from_generator(
        generate_training_example, (tf.int64, tf.int64),
        (tf.TensorShape([20, 24]), tf.TensorShape([])))
    examples = examples.batch(BATCH_SIZE).prefetch(16)
    model.fit(
        x=examples,
        epochs=10,
        steps_per_epoch=1000,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/tensorboard')])
    tf.saved_model.save(model, MODEL_PATH)


def main():
    model = create_model()
    train_model(model)


if __name__ == "__main__":
    main()
