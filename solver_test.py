'''Tests for solver.py '''
import unittest
from unittest import mock

import cube as cube_lib
import solver


class GreedySolveTest(unittest.TestCase):
    def test_already_solved(self):
        cube = cube_lib.Cube()
        model = mock.Mock()
        self.assertIsNone(solver.greedy_solve(model, cube, depth=1))

    def test_one_rotation_depth_one(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]
        self.assertEqual(solver.greedy_solve(model, cube, depth=1),
                         rotation.invert())

    def test_one_rotation_depth_two(self):
        cube = cube_lib.Cube()
        rotation = cube_lib.Rotation(cube_lib.Face.LEFT, is_clockwise=True)
        cube.rotate_face(rotation)

        model = mock.Mock()
        model.predict.return_value = [1.0]
        self.assertEqual(solver.greedy_solve(model, cube, depth=2),
                         rotation.invert())

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
        cube.rotate_face(solver.greedy_solve(model, cube, depth=2))
        self.assertFalse(cube.is_solved())
        cube.rotate_face(solver.greedy_solve(model, cube, depth=2))
        self.assertTrue(cube.is_solved())


if __name__ == '__main__':
    unittest.main()
