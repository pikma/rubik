'''Solving the cube given a value model.'''

import collections
import dataclasses

from typing import Callable, Optional

import pandas as pd
import tensorflow as tf

import cube as cube_lib
import util


@dataclasses.dataclass
class _Trajectory:
    final_state: cube_lib.Cube
    first_rotation: Optional[cube_lib.Rotation] = None
    num_rotations: int = 0


class GreedySolver:
    '''A solver that uses a greedy algorithm.

    It looks ahead in the tree of future states up to a given depth, evaluates
    states using the provided model, and picks the next rotation that gets us
    closer to solving the cube.
    '''
    def __init__(self, cube: cube_lib.Cube, model: tf.keras.Model, depth: int):
        '''
        The model is assumed to predict the distance towards the solved state,
        i.e. the number of rotations that will need to be applied in order to
        solve the cube (this is the opposive of a value function).
        '''
        self.cube = cube.copy()
        self._model_batcher = util.ModelBatcher(1024,
                                                model,
                                                feature_shape=(20, 24))
        self._depth = depth

    def apply_next_rotation(self):
        '''Applies one rotation to the cube.

        Returns the rotation that was applied, or None if the cube was already
        solved.
        '''
        if self.cube.is_solved():
            return None

        # We do a BFS so that we traverse states in increasing order of depth.
        # That way, as soon as we encounter a solved state, we know that we
        # have found the shortest path.
        queue = collections.deque()
        queue.append(_Trajectory(final_state=self.cube))

        explored_set = {self.cube}

        best_rotation = None
        best_value = None

        def process_predictions():
            nonlocal best_rotation
            nonlocal best_value
            for value, first_rotation in (
                    self._model_batcher.get_predictions()):
                value = value[0]
                # The model predicts the distance, so we minimize it.
                if best_value is None or value < best_value:
                    best_value = value
                    best_rotation = first_rotation

        while queue:
            trajectory = queue.pop()
            state = trajectory.final_state

            if state.is_solved():
                # We know this is the shortest trajectory since we are doing a
                # BFS.
                self.cube.rotate_face(trajectory.first_rotation)
                return trajectory.first_rotation

            if trajectory.num_rotations >= self._depth:
                # We reached a leaf in the tree, therefore we use the model to
                # evaluate the state.
                self._model_batcher.enqueue_predictions(
                    state.as_numpy_array().reshape((1, 20, 24)),
                    request_id=trajectory.first_rotation)

                process_predictions()
            else:
                # This isn't a leaf state, we expand it.
                for rotation in cube_lib.Rotation.all():
                    new_state = state.copy()
                    new_state.rotate_face(rotation)

                    if new_state not in explored_set:
                        explored_set.add(new_state)
                        # The first_rotation is set to None for empty
                        # trajectory (the one with the root state). If that's
                        # the case, then we are currently at the first
                        # rotation.
                        first_rotation = trajectory.first_rotation or rotation
                        queue.appendleft(
                            _Trajectory(
                                final_state=new_state,
                                first_rotation=first_rotation,
                                num_rotations=trajectory.num_rotations + 1))

        self._model_batcher.flush()
        process_predictions()

        self.cube.rotate_face(best_rotation)
        return best_rotation


def num_steps_to_solve(solver: GreedySolver, max_num_steps: int):
    '''Returns the number of steps required to solve a cube.

    Tries 'max_num_steps' steps, and returns None if the cube is not solved
    after that.
    '''
    for step in range(max_num_steps):
        if solver.cube.is_solved():
            return step
        solver.apply_next_rotation()

    if solver.cube.is_solved():
        return max_num_steps

    return None


def evaluate_solver(solver_fn: Callable[[cube_lib.Cube],
                                        GreedySolver], trajectory_length: int,
                    max_num_steps: int, num_trials: int):
    evaluation_results = pd.DataFrame()
    for _ in range(num_trials):
        cube = cube_lib.get_scrambled_cube(trajectory_length)
        solver = solver_fn(cube)

        num_steps = num_steps_to_solve(solver, max_num_steps)

        evaluation_results = evaluation_results.append(
            {
                'num_steps_scrambled': trajectory_length,
                'num_steps_to_solve': num_steps,
                'solved': solver.cube.is_solved()
            },
            ignore_index=True)
    evaluation_results.num_steps_scrambled = (
        evaluation_results.num_steps_scrambled.astype('int'))
    # We can't convert 'num_steps_to_solve' to ints, because it contains NAs,
    # which are of type float.
    evaluation_results.num_steps_to_solve = (
        evaluation_results.num_steps_to_solve.astype('float'))
    evaluation_results.solved = evaluation_results.solved.astype('bool')
    return evaluation_results


def main():
    model = tf.keras.models.load_model('./model')
    evaluation = evaluate_solver(lambda cube: GreedySolver(cube, model, 2),
                                 trajectory_length=10,
                                 max_num_steps=15,
                                 num_trials=20)
    print(evaluation)


if __name__ == "__main__":
    main()
