'''Tests for solver.py '''
import copy
import unittest
from unittest import mock

import numpy as np  # type: ignore

import cube as cube_lib
import solver as solver_lib


class FakeModel:
    def __init__(self, prediction: float):
        self._prediction = prediction

    def predict(self, features, *args, **kwargs):
        return self._prediction * np.ones((features.shape[0], 1))


class GreedySolveTest(unittest.TestCase):
    def test_already_solved_greedy(self):
        cube = cube_lib.Cube()
        model = mock.Mock()

        solver = solver_lib.GreedySolver(cube, model, depth=1)
        self.assertIsNone(solver.apply_next_rotation())

    def test_one_rotation_greedy_depth_one(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]

        solver = solver_lib.GreedySolver(cube, model, depth=1)
        self.assertEqual(solver.apply_next_rotation(), rotation.invert())
        self.assertTrue(solver.cube.is_solved())

    def test_one_rotation_greedy_depth_two(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]
        solver = solver_lib.GreedySolver(cube, model, depth=2)
        self.assertEqual(solver.apply_next_rotation(), rotation.invert())

    def test_two_rotations_greedy_depth_two(self):
        cube = cube_lib.Cube()
        cube.rotate_face(
            cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True))
        cube.rotate_face(
            cube_lib.Rotation(cube_lib.Face.UP, is_clockwise=False))

        model = mock.Mock()
        model.predict.return_value = [1.0]

        # We follow the solver for two steps, and verify that the cube is
        # solved.
        self.assertFalse(cube.is_solved())

        solver = solver_lib.GreedySolver(cube, model, depth=2)

        self.assertEqual(
            solver.apply_next_rotation(),
            cube_lib.Rotation(cube_lib.Face.UP, is_clockwise=True))
        self.assertFalse(solver.cube.is_solved())
        self.assertEqual(
            solver.apply_next_rotation(),
            cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=False))
        self.assertTrue(solver.cube.is_solved())

    def test_already_solved_astar(self):
        cube = cube_lib.Cube()
        model = mock.Mock()

        solver = solver_lib.AStarSolver(cube, model)
        self.assertIsNone(solver.apply_next_rotation())

    def test_one_rotation_a_star(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        # This is a valid heuristic, as it is an under-prediction. In that case,
        # AStar is equivalent to DFS.
        model = FakeModel(0)

        solver = solver_lib.AStarSolver(cube, model)
        self.assertEqual(solver.apply_next_rotation(), rotation.invert())

        self.assertTrue(solver.cube.is_solved())

    def test_priority_queue(self):
        queue = solver_lib._PriorityQueue()
        first_cube = cube_lib.get_scrambled_cube(3)

        first_state = solver_lib._AStarState(first_cube,
                                             cost_to_come=2,
                                             est_cost_to_go=5,
                                             previous_state=None,
                                             previous_rotation=None)
        queue.add_or_update_state(first_state)

        second_cube = first_cube.copy()
        second_cube.rotate_face(
            cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True))
        second_state = solver_lib._AStarState(second_cube,
                                              cost_to_come=1,
                                              est_cost_to_go=7,
                                              previous_state=None,
                                              previous_rotation=None)
        queue.add_or_update_state(second_state)

        third_cube = second_cube.copy()
        third_cube.rotate_face(
            cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True))
        third_state = solver_lib._AStarState(third_cube,
                                             cost_to_come=1,
                                             est_cost_to_go=5,
                                             previous_state=None,
                                             previous_rotation=None)
        queue.add_or_update_state(third_state)

        self.assertTrue(queue)
        self.assertEqual(queue.pop_min_state(), third_state)

        second_state = copy.copy(second_state)
        second_state.est_cost_to_go = 3
        queue.add_or_update_state(second_state)

        self.assertTrue(queue)
        self.assertEqual(queue.pop_min_state(), second_state)

        first_state = copy.copy(first_state)
        first_state.est_cost_to_go = 3
        queue.add_or_update_state(first_state)

        self.assertTrue(queue)
        self.assertEqual(queue.pop_min_state(), first_state)

        self.assertFalse(queue)

    def test_priority_queue_same_value(self):
        '''Verifies that we can insert two states with the same value.'''
        queue = solver_lib._PriorityQueue()
        first_cube = cube_lib.get_scrambled_cube(num_rotations=3)

        first_state = solver_lib._AStarState(first_cube,
                                             cost_to_come=2,
                                             est_cost_to_go=5,
                                             previous_state=None,
                                             previous_rotation=None)
        queue.add_or_update_state(first_state)

        second_cube = first_cube.copy()
        second_cube.rotate_face(
            cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True))
        second_state = solver_lib._AStarState(second_cube,
                                              cost_to_come=2,
                                              est_cost_to_go=5,
                                              previous_state=None,
                                              previous_rotation=None)
        queue.add_or_update_state(second_state)

        self.assertTrue(queue)

        costs_to_come = set()
        costs_to_come.add(queue.pop_min_state().cube)
        self.assertTrue(queue)
        costs_to_come.add(queue.pop_min_state().cube)
        self.assertFalse(queue)

        self.assertEqual(costs_to_come, set((first_cube, second_cube)))


if __name__ == '__main__':
    unittest.main()
