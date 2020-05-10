'''Benchmarks of various parts of the code.'''
import time

import numpy as np

import cube as cube_lib
import solver as solver_lib
import trainer

NUM_EXAMPLES = 1024 * 8 * 4


def generate_supervised_value_examples():
    '''Loops through training examples.

    Results (2020/05/02): 3000 examples/s on my machine.
    '''
    examples = trainer.get_supervised_value_examples()
    examples_it = iter(examples)
    begin_time = time.monotonic()
    for _ in range(NUM_EXAMPLES):
        next(examples_it)
    end_time = time.monotonic()
    print('{:.1f} examples/s'.format(NUM_EXAMPLES / (end_time - begin_time)))


def generate_td_value_examples():
    '''Loops through training examples.

    Results (2020/05/20): 660 examples/s on my machine.
    '''
    model = trainer.create_model()
    examples = trainer.get_td_value_examples(model)
    examples_it = iter(examples)
    begin_time = time.monotonic()
    for _ in range(NUM_EXAMPLES):
        next(examples_it)
    end_time = time.monotonic()
    print('{:.1f} examples/s'.format(NUM_EXAMPLES / (end_time - begin_time)))


def model_inference():
    '''Runs the model.

    Results (2020/05/02): 8192 is the best batch size.
        85.7 examples/s (batch_size = 1)
        163.1 examples/s (batch_size = 2)
        344.8 examples/s (batch_size = 4)
        670.1 examples/s (batch_size = 8)
        1352.3 examples/s (batch_size = 16)
        2695.4 examples/s (batch_size = 32)
        5335.5 examples/s (batch_size = 64)
        10341.2 examples/s (batch_size = 128)
        19092.1 examples/s (batch_size = 256)
        34406.9 examples/s (batch_size = 512)
        56986.0 examples/s (batch_size = 1024)
        91354.9 examples/s (batch_size = 2048)
        129481.5 examples/s (batch_size = 4096)
        168470.2 examples/s (batch_size = 8192)
        148578.4 examples/s (batch_size = 16384)
    '''
    batch_size = 1
    while batch_size <= 16 * 1024:
        _model_inference_inner_loop(batch_size)
        batch_size *= 2


def _model_inference_inner_loop(batch_size):
    cube = cube_lib.Cube()
    features = cube.as_numpy_array()

    features_batched = np.tile(features, [batch_size, 1, 1])
    assert (features_batched.shape == (batch_size, 20, 24))

    model = trainer.create_model()

    begin_time = time.monotonic()
    num_runs = 1000
    for _ in range(num_runs):
        model.predict(features_batched, batch_size=batch_size)
    end_time = time.monotonic()
    print('{:.1f} examples/s (batch_size = {})'.format(
        num_runs * batch_size / (end_time - begin_time), batch_size))


def greedy_solver(depth=3):
    '''Runs the greedy solver.

    Results (2020/05/20):
        3.4 cubes solved / second at depth 3.
    '''
    model = trainer.create_model()
    num_starting_cubes = 20
    num_runs = 3

    num_runs_done = 0
    time_elapsed = 0
    for _ in range(num_starting_cubes):
        cube = cube_lib.get_scrambled_cube(depth)
        begin_time = time.monotonic()

        for _ in range(num_runs):
            solver = solver_lib.GreedySolver(cube, model, depth)
            num_rotations = 0
            while not solver.cube.is_solved():
                if num_rotations == depth:
                    raise Exception(
                        'Done {} rotations, cube still not solved'.format(
                            depth))
                solver.apply_next_rotation()
                num_rotations += 1

        end_time = time.monotonic()
        time_elapsed += (end_time - begin_time)
        num_runs_done += num_runs

    print('{} cubes solved per second at depth {}'.format(
        num_runs_done / time_elapsed, depth))


def main():
    greedy_solver(depth=3)


if __name__ == "__main__":
    main()
