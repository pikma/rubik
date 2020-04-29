from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import cube as cube_lib
import solver as solver_lib

HIDDEN_LAYERS_WIDTH = [4096, 2048, 1024]


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

BATCH_SIZE = 32
MODEL_PATH = './models'

NUM_EPOCHS = 10
NUM_STEPS_PER_EPOCH = 1000


def train_supervised_value_model(model: tf.keras.Model) -> None:
    '''Takes a compiled model and trains it.

    Trains a value model in a supervised mannel: it simply tries to predict the
    value at each state.
    '''
    def generated_supervised_value_examples() -> Tuple[np.ndarray, int]:
        '''Generates training examples.'''
        while True:
            for i, cube in enumerate(
                    cube_lib.scramble_cube(TRAJECTORY_LENGTH)):
                yield (cube.as_numpy_array(), i + 1)

    examples = tf.data.Dataset.from_generator(
        generated_supervised_value_examples, (tf.int64, tf.int64),
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
            steps_per_epoch=NUM_STEPS_PER_EPOCH,
            initial_epoch=epoch,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir='/tmp/tensorboard')
            ])
        epoch_evaluation_df = evaluate_model(model)
        print(epoch_evaluation_df)
        epoch_evaluation_df['epoch'] = epoch
        evaluation_df = evaluation_df.append(epoch_evaluation_df,
                                             ignore_index=True)
    tf.saved_model.save(model, './models/supervised_value')


def train_td_value_model(model: tf.keras.Model) -> None:
    '''Trains a compiled model.

    The model is trained using TD-learning: it tries to predict 1 plus the
    maximum of its own prediction on the successor states.
    '''
    # TODO: Right now, this is extremely slow (~1 step/sec). The generation of
    # examples is the slow part, one way to make it faster is to batch the
    # predictions more.
    def generate_td_value_examples() -> Tuple[np.ndarray, float]:
        '''Generates training examples.'''
        while True:
            for cube in cube_lib.scramble_cube(TRAJECTORY_LENGTH):
                next_value = None
                next_cube_features = []

                for rotation in cube_lib.Rotation.all():
                    next_cube = cube.copy()
                    next_cube.rotate_face(rotation)
                    if next_cube.is_solved():
                        next_value = 1
                    else:
                        next_cube_features.append(next_cube.as_numpy_array())

                if next_value is None:
                    next_cube_predictions = model.predict([next_cube_features])
                    assert next_cube_predictions.shape == (
                        12, 1), next_cube_predictions.shape
                    next_value = 1 + np.min(next_cube_predictions)

                yield (cube.as_numpy_array(), next_value)

    # TODO: we generate the labels using the model that we train. Consider
    # keeping the label model fixed for a while until the trained model
    # converges, and only then updating it.
    examples = tf.data.Dataset.from_generator(
        generate_td_value_examples, (tf.int64, tf.float32),
        (tf.TensorShape([20, 24]), tf.TensorShape([])))
    examples = examples.shuffle(
        buffer_size=4096).batch(BATCH_SIZE).prefetch(16)

    evaluation_df = pd.DataFrame()
    for epoch in range(NUM_EPOCHS):
        model.fit(
            x=examples,
            epochs=epoch + 1,  # Train for one epoch.
            steps_per_epoch=NUM_STEPS_PER_EPOCH,
            initial_epoch=epoch,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir='/tmp/tensorboard')
            ])
        epoch_evaluation_df = evaluate_model(model)
        print(epoch_evaluation_df)
        epoch_evaluation_df['epoch'] = epoch
        evaluation_df = evaluation_df.append(epoch_evaluation_df,
                                             ignore_index=True)
    tf.saved_model.save(model, './models/td_value')


def fraction_solved_greedy(model: tf.keras.Model, trajectory_length: int,
                           num_trials: int, greedy_depth: int) -> float:
    '''Computes the fraction of random cubes that can be solved greedily.'''
    num_solved = 0
    for _ in range(num_trials):
        cube = cube_lib.get_scrambled_cube(trajectory_length)
        solver = solver_lib.GreedySolver(cube, model, depth=greedy_depth)
        for _ in range(trajectory_length):
            if solver.cube.is_solved():
                break
            solver.apply_next_rotation()

        if solver.cube.is_solved():
            num_solved += 1
    return num_solved / num_trials


def evaluate_model(model: tf.keras.Model) -> pd.DataFrame:
    '''Evaluates a model's performance.'''
    evaluation_results = pd.DataFrame()
    for trajectory_length in [2, 5, 10]:
        fraction_solved = fraction_solved_greedy(
            model,
            trajectory_length=trajectory_length,
            num_trials=10,
            greedy_depth=1)
        evaluation_results = evaluation_results.append(
            {
                'trajectory_length': trajectory_length,
                'fraction_solved': fraction_solved,
            },
            ignore_index=True)
    return evaluation_results


def main():
    model = create_model()
    #  train_supervised_value_model(model)
    train_td_value_model(model)


if __name__ == "__main__":
    main()
