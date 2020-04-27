''' Tests for cube.py '''

import random
import unittest

import cube as cube_lib


class CubeTest(unittest.TestCase):
    ''' Tests for the Cube class. '''
    def test_rotations_four_times(self):
        '''Applying the same rotation four times leaves the cube unchanged.'''
        for rotation in cube_lib.Rotation.all():
            cube = cube_lib.Cube()
            for _ in range(4):
                cube.rotate_face(rotation)
            self.assertEqual(cube,
                             cube_lib.Cube(),
                             msg='Rotation is {}'.format(rotation))

    def test_random_rotations_and_back(self):
        '''Doing a rotation and then its invert leaves the cube unchanged.'''
        random.seed(0)

        rotations = []

        for _ in range(20):
            rotations.append(cube_lib.Rotation.random_new())

        cube = cube_lib.Cube()
        for rotation in rotations:
            cube.rotate_face(rotation)

        for rotation in reversed(rotations):
            cube.rotate_face(rotation.invert())

        self.assertEqual(cube, cube_lib.Cube())

    def test_scrambled_cube(self):
        '''Scrambling the cube with cannot return a solved cube.

        This is guaranteed for 2 and 3 rotations, because scramble_cube does
        not apply two consecutive rotations that cancel each other.
        '''
        self.assertNotEqual(cube_lib.get_scrambled_cube(num_rotations=2),
                            cube_lib.Cube())
        self.assertNotEqual(cube_lib.get_scrambled_cube(num_rotations=3),
                            cube_lib.Cube())


if __name__ == '__main__':
    unittest.main()
