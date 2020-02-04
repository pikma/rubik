import random
import unittest

import cube as cube_lib


class CubeTest(unittest.TestCase):
    def test_rotations_four_times(self):
        for face in [cube_lib.Face(i) for i in range(cube_lib.NUM_FACES)]:
            for clockwise in (False, True):
                cube = cube_lib.Cube()
                for i in range(4):
                    cube.rotate_face(face, clockwise)
                self.assertEqual(cube,
                                 cube_lib.Cube(),
                                 msg='Face is {}, clockwise is {}'.format(
                                     face, clockwise))

    def test_random_rotations_and_back(self):
        random.seed(0)

        rotations = []

        for _ in range(20):
            face = cube_lib.Face(random.randrange(0, cube_lib.NUM_FACES))
            clockwise = random.random() > 0.5
            rotations.append((face, clockwise))

        cube = cube_lib.Cube()
        for face, clockwise in rotations:
            cube.rotate_face(face, clockwise)

        for face, clockwise in reversed(rotations):
            cube.rotate_face(face, not clockwise)

        self.assertEqual(cube, cube_lib.Cube())


if __name__ == '__main__':
    unittest.main()
