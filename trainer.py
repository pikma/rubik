from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import cube as cube_lib
import solver

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


TRAJECTORY_LENGTH = 32


def generate_training_example() -> Tuple[np.ndarray, int]:
    '''Generates training examples.'''
    while True:
        cube = cube_lib.Cube()
        for i in range(TRAJECTORY_LENGTH):
            cube.rotate_face(cube_lib.Rotation.random_new())
            yield (cube.as_numpy_array(), i)


BATCH_SIZE = 32
MODEL_PATH = './model'

NUM_EPOCHS = 10


def train_model(model: tf.keras.Model) -> None:
    '''Takes a compiled model and trains it.'''
    examples = tf.data.Dataset.from_generator(
        generate_training_example, (tf.int64, tf.int64),
        (tf.TensorShape([20, 24]), tf.TensorShape([])))
    # Training examples are generated from trajectories, so consecutive
    # examples are strongly correlated. This increases the variance of the
    # gradient. Shuffling the examples reduces the variance and speeds up
    # training significantly.
    examples = examples.shuffle(
        buffer_size=4096).batch(BATCH_SIZE).prefetch(16)

    evaluation_df = pd.DataFrame()
    for epoch in range(NUM_EPOCHS):
        model.fit(
            x=examples,
            epochs=epoch + 1,  # Train for one epoch.
            steps_per_epoch=10000,
            initial_epoch=epoch,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir='/tmp/tensorboard')
            ])
        epoch_evaluation_df = evaluate_model(model)
        print(epoch_evaluation_df)
        epoch_evaluation_df['epoch'] = epoch
        evaluation_df = evaluation_df.append(epoch_evaluation_df,
                                             ignore_index=True)
    tf.saved_model.save(model, MODEL_PATH)


def fraction_solved_greedy(model: tf.keras.Model, trajectory_length: int,
                           num_trials: int, greedy_depth: int) -> float:
    '''Computes the fraction of random cubes that can be solved greedily.'''
    num_solved = 0
    for _ in range(num_trials):
        cube = cube_lib.Cube()
        for _ in range(trajectory_length):
            rotation = cube_lib.Rotation.random_new()
            cube.rotate_face(rotation)

        for _ in range(trajectory_length):
            if cube.is_solved():
                break
            rotation = solver.greedy_solve(model, cube, depth=greedy_depth)
            cube.rotate_face(rotation)

        if cube.is_solved():
            num_solved += 1
    return num_solved / num_trials


def evaluate_model(model: tf.keras.Model) -> pd.DataFrame:
    '''Evaluates a model's performance.'''
    df = pd.DataFrame()
    for trajectory_length in [2, 5, 10]:
        fraction_solved = fraction_solved_greedy(
            model,
            trajectory_length=trajectory_length,
            num_trials=10,
            greedy_depth=1)
        df = df.append(
            {
                'trajectory_length': trajectory_length,
                'fraction_solved': fraction_solved,
            },
            ignore_index=True)
    return df


def main():
    model = create_model()
    train_model(model)


if __name__ == "__main__":
    main()
