'''Tests for solver.py '''
import unittest
from unittest import mock

import cube as cube_lib
import solver as solver_lib


class GreedySolveTest(unittest.TestCase):
    def test_already_solved(self):
        cube = cube_lib.Cube()
        model = mock.Mock()

        solver = solver_lib.GreedySolver(cube, model, depth=1)
        self.assertIsNone(solver.apply_next_rotation())

    def test_one_rotation_depth_one(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]

        solver = solver_lib.GreedySolver(cube, model, depth=1)
        self.assertEqual(solver.apply_next_rotation(), rotation.invert())

    def test_one_rotation_depth_two(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]
        solver = solver_lib.GreedySolver(cube, model, depth=2)
        self.assertEqual(solver.apply_next_rotation(), rotation.invert())

    def test_two_rotations_depth_two(self):
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


if __name__ == '__main__':
    unittest.main()
