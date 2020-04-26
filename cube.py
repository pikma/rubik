'''A representation of a Rubik's cube.'''

import collections
import enum
import random
import typing

from typing import Dict, List, Tuple

import numpy as np

_RESET_ESCAPE_SEQUENCE = '\u001b[0m'


@enum.unique
class Color(enum.Enum):
    '''The six colors on the cube.'''
    WHITE = 0
    RED = 1
    BLUE = 2
    ORANGE = 3
    GREEN = 4
    YELLOW = 5

    def __lt__(self, other):
        return self.value < other.value

    def to_escape_sequence(self):
        ''' Returns the bash escape sequence corresponding to this color. '''
        code = {
            Color.WHITE: 15,
            Color.RED: 1,
            Color.BLUE: 4,
            Color.ORANGE: 208,
            Color.GREEN: 2,
            Color.YELLOW: 3
        }[self]
        return '\u001b[48;5;{}m'.format(code)


def _color_row_to_str(color_row: List[Color]) -> str:
    escape_sequences = [color.to_escape_sequence() for color in color_row]
    return '{}  {}  {}  {}'.format(escape_sequences[0], escape_sequences[1],
                                   escape_sequences[2], _RESET_ESCAPE_SEQUENCE)


@enum.unique
class Face(enum.Enum):
    '''A face of the Cube. This is the part that can be rotated.'''
    UP = 0  # pylint: disable=C0103
    LEFT = 1
    FRONT = 2
    RIGHT = 3
    BACK = 4
    DOWN = 5

    def __lt__(self, other):
        return self.value < other.value


_NUM_FACES = 6


class Rotation(typing.NamedTuple):
    '''A rotation of a face, that can be applied to the cube.'''
    face: Face
    is_clockwise: bool

    def invert(self):
        ''' Returns the rotation which is the invert of 'self'. '''
        return Rotation(self.face, not self.is_clockwise)

    @classmethod
    def random_new(cls):
        ''' Returns a random new Rotation. '''
        face = Face(random.randrange(0, _NUM_FACES))
        is_clockwise = random.random() > 0.5
        return cls(face, is_clockwise)

    @classmethod
    def all(cls):
        ''' Yields all possible rotations. '''
        for face in [Face(i) for i in range(_NUM_FACES)]:
            for is_clockwise in (False, True):
                yield cls(face, is_clockwise)


def _init_block_to_pos():
    # The cube is represented internally as a one-hot 20 * 24 matrix. Each
    # row represents a block (corner or edge), and for each row, exactly
    # one column is set to 1.  There are 24 possible positions for each
    # block (for angles, 8 positions * 3 orientations; for edges, 12
    # positions * 2 orientations), therefore the column that is set to 1
    # represents the position and orientation of the block. See
    # _POSITION_TO_INDEX and _BLOCK_TO_INDEX to understand the indices.

    # This idea comes from McAleer 2018 (https://arxiv.org/abs/1805.07470).
    result = np.zeros([20, 24], np.int)

    # Corners:
    result[0, 0] = 1
    result[1, 3] = 1
    result[2, 6] = 1
    result[3, 9] = 1
    result[4, 12] = 1
    result[5, 15] = 1
    result[6, 18] = 1
    result[7, 21] = 1
    result[8, 0] = 1
    result[9, 2] = 1
    result[10, 4] = 1
    result[11, 6] = 1
    result[12, 8] = 1
    result[13, 10] = 1
    result[14, 12] = 1
    result[15, 14] = 1
    result[16, 16] = 1
    result[17, 18] = 1
    result[18, 20] = 1
    result[19, 22] = 1
    return result


_SOLVED_BLOCK_TO_POS = _init_block_to_pos()


class Cube:
    ''' A Rubik's cube.

    '''

    def __init__(self):
        self._block_to_pos = _SOLVED_BLOCK_TO_POS.copy()

    def is_solved(self):
        ''' Returns true if the cube is solved. '''
        return np.all(self._block_to_pos == _SOLVED_BLOCK_TO_POS)

    def _get_face_colors(self, face):
        face_neighbors = _get_face_neighbors(face)
        positions = [(face, face_neighbors.top, face_neighbors.left),
                     (face, face_neighbors.top),
                     (face, face_neighbors.right, face_neighbors.top),
                     (face, face_neighbors.left), (face, face_neighbors.right),
                     (face, face_neighbors.left, face_neighbors.bottom),
                     (face, face_neighbors.bottom),
                     (face, face_neighbors.bottom, face_neighbors.right)]
        positions = map(_normalize_position, positions)
        colors = [self._get_color(position, face) for position in positions]
        colors.insert(4, _get_middle_color(face))
        colors = [colors[0:3], colors[3:6], colors[6:9]]
        return colors

    def _get_color(self, position: Tuple[Face, ...], face: Face) -> Color:
        base_pos_index = _POSITION_TO_INDEX[position]

        # Here we search for the block which is at the given position. Since
        # corners and edges are stored in different rows of the _block_to_pos
        # array, we need to search separately for both cases.
        if len(position) == 2:
            block_to_pos = self._block_to_pos[8:,
                                              base_pos_index:base_pos_index +
                                              2]
        elif len(position) == 3:
            block_to_pos = self._block_to_pos[:8,
                                              base_pos_index:base_pos_index +
                                              3]

        # Here we need to take into account the fact that blocks at a given
        # position can be rotated in different ways. The block_index is the
        # index of the block at the given location, and rotation is in {0, 1}
        # for edges, and {0, 1, 2} for corners: it is the index, in 'position',
        # of the face where the first color of 'block is.
        block_index, rotation = np.argwhere(block_to_pos)[0]

        if len(position) == 2:
            # Because we searched for block_index in the slice starting at 8.
            block_index += 8

        block = _INDEX_TO_BLOCK[block_index]
        face_ix = position.index(face)
        color = block[(face_ix - rotation) % len(block)]

        return color

    def __str__(self) -> str:
        faces = (Face(i) for i in range(_NUM_FACES))
        face_colors = {face: self._get_face_colors(face) for face in faces}
        lines = []

        for row in face_colors[Face.UP]:
            lines.append('      ' + _color_row_to_str(row))

        for row_ix in range(3):
            line = ''
            for face in [Face.LEFT, Face.FRONT, Face.RIGHT, Face.BACK]:
                line += _color_row_to_str(face_colors[face][row_ix])
            lines.append(line)

        for row in face_colors[Face.DOWN]:
            lines.append('      ' + _color_row_to_str(row))
        return '\n'.join(lines)

    def _rotate_blocks(self, from_to_rotations: Tuple[int, int, int],
                       corners: bool):
        # We save the last position slice, and then apply the moves in reverse
        # order, so that the last move reads from the saved position slice.
        last_index = from_to_rotations[0][0]
        if corners:
            max_rotation = 3
            blocks_to_pos = self._block_to_pos[:8]
        else:
            max_rotation = 2
            blocks_to_pos = self._block_to_pos[8:]

        tmp = blocks_to_pos[:, last_index:last_index + max_rotation].copy()

        for from_ix, to_ix, rotation in reversed(from_to_rotations):
            if from_ix == last_index:
                old = tmp
            else:
                old = blocks_to_pos[:, from_ix:from_ix + max_rotation]
            old = np.roll(old, rotation, axis=1)
            blocks_to_pos[:, to_ix:to_ix + max_rotation] = old

    def _print_debug_state(self):
        print('--0 1 2 3 4 5 6 7 8 9 . 1 2 3 4 5 6 7 8 9 . 1 2 3')
        print(self._block_to_pos[:8])
        print('--0 1 2 3 4 5 6 7 8 9 . 1 2 3 4 5 6 7 8 9 . 1 2 3')
        print(self._block_to_pos[8:])
        print('--0 1 2 3 4 5 6 7 8 9 . 1 2 3 4 5 6 7 8 9 . 1 2 3')
        print()

    def _rotate_corners(self, from_to_rotations: Tuple[int, int, int]):
        self._rotate_blocks(from_to_rotations, corners=True)

    def _rotate_edges(self, from_to_rotations: Tuple[int, int, int]):
        self._rotate_blocks(from_to_rotations, corners=False)

    def rotate_face(self, rotation: Rotation) -> None:
        self._check_blocks_to_pos()

        # Each value in the list is the rotation to apply to the corners, and
        # the rotation to apply to the edges (in a pair). Each rotation is
        # represented as a list of (from_ix, to_ix, rotation): the block at
        # position indexed by from_ix needs to be moved to the position index
        # by to_ix, with a rotation applied.
        rotations = {
            Face.UP: (
                [(0, 3, 0), (3, 6, 0), (6, 9, 0), (9, 0, 0)],  #
                [(0, 2, 0), (2, 4, 0), (4, 6, 0), (6, 0, 0)]),
            Face.LEFT: (
                [(9, 0, 2), (0, 12, 2), (12, 21, 0), (21, 9, 2)],  #
                [(0, 8, 1), (8, 22, 0), (22, 14, 0), (14, 0, 1)]),
            Face.FRONT: (
                [(0, 3, 2), (3, 15, 2), (15, 12, 2), (12, 0, 0)],  #
                [(2, 10, 1), (10, 16, 0), (16, 8, 1), (8, 2, 0)]),
            Face.RIGHT: (
                [(3, 6, 2), (6, 18, 2), (18, 15, 2), (15, 3, 0)],  #
                [(4, 12, 1), (12, 18, 0), (18, 10, 1), (10, 4, 0)]),
            Face.BACK: (
                [(6, 9, 2), (9, 21, 0), (21, 18, 1), (18, 6, 0)],  #
                [(6, 14, 0), (14, 20, 1), (20, 12, 1), (12, 6, 0)]),
            Face.DOWN: (
                [(12, 15, 0), (15, 18, 0), (18, 21, 1), (21, 12, 2)],  #
                [(16, 18, 0), (18, 20, 0), (20, 22, 0), (22, 16, 0)]),
        }
        corner_rotations, edge_rotations = rotations[rotation.face]

        def invert_rotation(rotation):
            return [(to_ix, from_ix, -rot)
                    for from_ix, to_ix, rot in reversed(rotation)]

        if not rotation.is_clockwise:
            corner_rotations = invert_rotation(corner_rotations)
            edge_rotations = invert_rotation(edge_rotations)

        self._rotate_corners(corner_rotations)
        self._rotate_edges(edge_rotations)

    def _check_blocks_to_pos(self) -> None:
        #  self._print_debug_state()
        # All cells are 0 or 1.
        assert np.all((self._block_to_pos == 0.0) | (self._block_to_pos == 1))

        # Each block has exactly one position.
        num_pos_per_block = self._block_to_pos.sum(axis=1)
        assert np.all(num_pos_per_block == 1)

        # Each position is filled by one corner block.
        splits = np.split(self._block_to_pos[:8], _NUM_POSITIONS // 3, axis=1)
        for i, pos in enumerate(splits):
            assert pos.sum() == 1, 'Position {}'.format(3 * i)

        # Each position is filled by one edge block.
        splits = np.split(self._block_to_pos[8:], _NUM_POSITIONS // 2, axis=1)
        for i, pos in enumerate(splits):
            assert pos.sum() == 1, 'Position {}'.format(2 * i)

    def __eq__(self, other):
        return np.all(self._block_to_pos == other._block_to_pos)

    def __hash__(self):
        num_rows, _ = self._block_to_pos.shape
        as_row = (self._block_to_pos *
                  np.arange(num_rows).reshape([num_rows, 1])).sum(axis=0)
        return tuple(as_row).__hash__()

    def as_numpy_array(self) -> np.array:
        return self._block_to_pos.copy()

    def copy(self):
        new_cube = Cube()
        new_cube._block_to_pos = self._block_to_pos.copy()
        return new_cube


# The blocks need to map to the positions in _POSITION_TO_INDEX, so that
# initializing the cube in its solved state is trivial.
_BLOCK_TO_INDEX: Dict[Tuple[Color, ...], int] = {
    (Color.WHITE, Color.RED, Color.BLUE): 0,
    (Color.WHITE, Color.BLUE, Color.ORANGE): 1,
    (Color.WHITE, Color.ORANGE, Color.GREEN): 2,
    (Color.WHITE, Color.GREEN, Color.RED): 3,
    (Color.RED, Color.YELLOW, Color.BLUE): 4,
    (Color.BLUE, Color.YELLOW, Color.ORANGE): 5,
    (Color.ORANGE, Color.YELLOW, Color.GREEN): 6,
    (Color.RED, Color.GREEN, Color.YELLOW): 7,
    (Color.WHITE, Color.RED): 8,
    (Color.WHITE, Color.BLUE): 9,
    (Color.WHITE, Color.ORANGE): 10,
    (Color.WHITE, Color.GREEN): 11,
    (Color.RED, Color.BLUE): 12,
    (Color.BLUE, Color.ORANGE): 13,
    (Color.ORANGE, Color.GREEN): 14,
    (Color.RED, Color.GREEN): 15,
    (Color.BLUE, Color.YELLOW): 16,
    (Color.ORANGE, Color.YELLOW): 17,
    (Color.GREEN, Color.YELLOW): 18,
    (Color.RED, Color.YELLOW): 19,
}

_INDEX_TO_BLOCK = {v: k for k, v in _BLOCK_TO_INDEX.items()}

_NUM_BLOCKS = len(_BLOCK_TO_INDEX)


def _normalize_position(position: Tuple[Face, ...]) -> Tuple[Face, ...]:
    '''Normalizes a position (edge or corner).

    For corner, it is assumed that the three faces are listed in
    counter-clockwise order when looking at the cube.
    '''
    min_value = None
    for i, face in enumerate(position):
        if min_value is None or face.value < min_value:
            min_value = face.value
            min_face_ix = i

    return position[min_face_ix:] + position[:min_face_ix]


# Corners can be oriented in three different ways. To encode the orientation,
# we look at where the first color in the block tuple is. If it is on the first
# face in the position tuple (for "Up Left Front", Up), then the index in the
# array is 0 mod 3. If it is on the second face (Left), then it is 1 mod 3, and
# if it is on the third face (Front), then it is 2 mod 3.
#
# For a position tuple and a corner tuple, the color corner[0] is on the face
# tuple[orientation]. Equivalently, the color on face with index face_ix is
# face_ix - rotation.
#
# For this to work, the cubes need to all be in the same geometrical order,
# which is counter-clockwise when looking at the corner itself.
#
# We do the same for edges: they have 2 orientations, so we use the mod 2 to
# encode the orientation.
#
# Note: one problem is that each corner could in theory be represented by the
# three faces it contains, in any order. That makes it impossible to loop up a
# corner in the map without trying all possibilities. To solve this, we
# represent the three faces in each corner in a consistent order:
# geometrically, we start with the face that has the lowest enum value, and
# then we list the two other edges in counter-clockwise order (when holding the
# cube in front of you, facing the corner). For edges, it is simpler, we only
# need to list the face with the lowest enum value first. Use
# _normalize_position to do this when looking up a position.
_POSITION_TO_INDEX: Dict[Tuple[Face, ...], int] = {
    # Corners:
    (Face.UP, Face.LEFT, Face.FRONT): 0,
    (Face.UP, Face.FRONT, Face.RIGHT): 3,
    (Face.UP, Face.RIGHT, Face.BACK): 6,
    (Face.UP, Face.BACK, Face.LEFT): 9,
    (Face.LEFT, Face.DOWN, Face.FRONT): 12,
    (Face.FRONT, Face.DOWN, Face.RIGHT): 15,
    (Face.RIGHT, Face.DOWN, Face.BACK): 18,
    (Face.LEFT, Face.BACK, Face.DOWN): 21,
    # Edges
    (Face.UP, Face.LEFT): 0,
    (Face.UP, Face.FRONT): 2,
    (Face.UP, Face.RIGHT): 4,
    (Face.UP, Face.BACK): 6,
    (Face.LEFT, Face.FRONT): 8,
    (Face.FRONT, Face.RIGHT): 10,
    (Face.RIGHT, Face.BACK): 12,
    (Face.LEFT, Face.BACK): 14,
    (Face.FRONT, Face.DOWN): 16,
    (Face.RIGHT, Face.DOWN): 18,
    (Face.BACK, Face.DOWN): 20,
    (Face.LEFT, Face.DOWN): 22,
}

for __position in _POSITION_TO_INDEX:
    assert __position == _normalize_position(__position)

_NUM_POSITIONS = 24

FaceNeighbors = collections.namedtuple('FaceNeighbors',
                                       ['left', 'top', 'right', 'bottom'])


def _get_face_neighbors(face: Face) -> FaceNeighbors:
    '''Returns the four faces that are neighbors to a given face.'''
    if face == Face.UP:
        return FaceNeighbors(left=Face.LEFT,
                             top=Face.BACK,
                             right=Face.RIGHT,
                             bottom=Face.FRONT)
    if face == Face.LEFT:
        return FaceNeighbors(left=Face.BACK,
                             top=Face.UP,
                             right=Face.FRONT,
                             bottom=Face.DOWN)
    if face == Face.FRONT:
        return FaceNeighbors(left=Face.LEFT,
                             top=Face.UP,
                             right=Face.RIGHT,
                             bottom=Face.DOWN)
    if face == Face.RIGHT:
        return FaceNeighbors(left=Face.FRONT,
                             top=Face.UP,
                             right=Face.BACK,
                             bottom=Face.DOWN)
    if face == Face.BACK:
        return FaceNeighbors(left=Face.RIGHT,
                             top=Face.UP,
                             right=Face.LEFT,
                             bottom=Face.DOWN)
    if face == Face.DOWN:
        return FaceNeighbors(left=Face.LEFT,
                             top=Face.FRONT,
                             right=Face.RIGHT,
                             bottom=Face.BACK)
    raise ValueError(face)


def _get_middle_color(face: Face) -> Color:
    return Color(face.value)


def get_scrambled_cube(num_rotations: int) -> Cube:
    '''Scrambles a cube with 'num_rotations' random rotations.'''
    cube = Cube()
    last_rotation = None
    for _ in range(num_rotations):
        # Make sure that we don't invert the previous rotation.
        next_rotation = None
        while next_rotation is None or last_rotation is None or next_rotation == last_rotation.invert():
            next_rotation = Rotation.random_new()
            last_rotation = next_rotation

        cube.rotate_face(next_rotation)

    return cube


def main():
    cube = Cube()
    print(cube)

    for _ in range(4):
        cube.rotate_face(Rotation(Face.DOWN, is_clockwise=True))
        print()
        print(cube)


if __name__ == "__main__":
    main()
