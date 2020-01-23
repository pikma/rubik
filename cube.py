'''A representation of a Rubik's cube.'''

import collections
import enum
from typing import Dict, List, Tuple

import numpy as np

RESET_ESCAPE_SEQUENCE = '\u001b[0m'


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
                                   escape_sequences[2], RESET_ESCAPE_SEQUENCE)


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


class Cube:
    ''' A Rubik's cube.

    It is represented internally as a one-hot 20 * 24 matrix. Each row
    represents a block (corner or edge), and for each row, exactly one column
    is set to 1.  There are 24 possible positions for each block (for angles, 8
    positions * 3 orientations; for edges, 12 positions * 2 orientations),
    therefore the column that is set to 1 represents the position and
    orientation of the block.

    This idea comes from McAleer 2018 (https://arxiv.org/abs/1805.07470).
    '''
    def __init__(self):
        # Corners:
        #   0: White Red Blue
        #   1: White Blue Orange
        #   2: White Orange Green
        #   3: White Red Green
        #   4: Red Blue Yellow
        #   5: Blue Orange Yellow
        #   6: Orange Green Yellow
        #   7: Red Green Yellow
        #
        # Angles:
        #   8: White Red
        #   9: White Blue
        #   10: White Orange
        #   11: White Green
        #   12: Red Blue
        #   13: Blue Orange
        #   14: Orange Green
        #   15: Red Green
        #   16: Blue Yellow
        #   17: Orange Yellow
        #   18: Green Yellow
        #   19: Red Yellow
        #
        # Positions for corner:
        #   Each position is determined by the faces that the block is on.
        #
        #   For each entry in the list below, there are three possible
        #   orientations of the block. To encode the orientation, we look at
        #   where the sticker with the first color (in the order of the Color
        #   enum) is. If it is on the first face in the list given below (for
        #   "Up Left Front", Up), then the index in the array is 0 mod 3. If it
        #   is on the second face (Left), then it is 1 mod 3, and if it is on
        #   the third face (Front), then it is 2 mod 3.
        #
        #   0-2: Up Left Front
        #   3-5: Up Front Right
        #   6-8: Up Right Back
        #   9-11: Up Left Back
        #   12-14: Left Front Down
        #   15-17: Front Right Down
        #   18-20: Right Back Down
        #   21-23: Left Back Down
        #
        # Positions for edge:
        #   For each, the position is 0 mod 2 if the edge's first color (in the
        #   order of the Color enum) is on the first listed below, and 1 mod 2
        #   if it
        #   is on the second face.
        #   0-1: Up Left
        #   2-3: Up Front
        #   4-5: Up Right
        #   6-7: Up Back
        #   8-9: Left Front
        #   10-11: Front Right
        #   12-13: Right Back
        #   14-15: Back Left
        #   16-17: Front Down
        #   18-19: Right Down
        #   20-21: Back Down
        #   22-23: Left Down
        self.block_to_pos = np.zeros([20, 24])

        # Corners:
        self.block_to_pos[0, 0] = 1
        self.block_to_pos[1, 3] = 1
        self.block_to_pos[2, 6] = 1
        self.block_to_pos[3, 9] = 1
        self.block_to_pos[4, 12] = 1
        self.block_to_pos[5, 15] = 1
        self.block_to_pos[6, 18] = 1
        self.block_to_pos[7, 21] = 1
        # Edges:
        self.block_to_pos[8, 0] = 1
        self.block_to_pos[9, 2] = 1
        self.block_to_pos[10, 4] = 1
        self.block_to_pos[11, 6] = 1
        self.block_to_pos[12, 8] = 1
        self.block_to_pos[13, 10] = 1
        self.block_to_pos[14, 12] = 1
        self.block_to_pos[15, 14] = 1
        self.block_to_pos[16, 16] = 1
        self.block_to_pos[17, 18] = 1
        self.block_to_pos[18, 20] = 1
        self.block_to_pos[19, 22] = 1

    def __get_face_colors(self, face):
        face_neighbors = _get_face_neighbors(face)
        positions = [(face, face_neighbors.left, face_neighbors.top),
                     (face, face_neighbors.top),
                     (face, face_neighbors.top, face_neighbors.right),
                     (face, face_neighbors.left), (face, face_neighbors.right),
                     (face, face_neighbors.left, face_neighbors.bottom),
                     (face, face_neighbors.bottom),
                     (face, face_neighbors.right, face_neighbors.bottom)]
        positions = map(sorted, positions)
        positions = map(tuple, positions)
        colors = [self.__get_color(position, face) for position in positions]
        colors.insert(4, _get_middle_color(face))
        colors = [colors[0:3], colors[3:6], colors[6:9]]
        return colors

    def __get_color(self, position: Tuple[Face, ...], face: Face) -> Color:
        base_pos_index = POSITION_TO_INDEX[position]

        # Here we search for the block which is at the given position. Since
        # corners and edges are stored in different rows of the block_to_pos
        # array, we need to search separately for both cases.
        if len(position) == 2:
            block_to_pos = self.block_to_pos[8:,
                                             base_pos_index:base_pos_index + 2]
        elif len(position) == 3:
            block_to_pos = self.block_to_pos[:8,
                                             base_pos_index:base_pos_index + 3]

        # Here we need to take into account the fact that blocks at a given
        # position can be rotated in different ways. The block_index is the
        # index of the block at the given location, and rotation is in {0, 1}
        # for edges, and {0, 1, 2} for corners: it is the index, in 'position',
        # of the face where the first color of 'block is.
        block_index, rotation = np.argwhere(block_to_pos)[0]

        if len(position) == 2:
            # Because we searched for block_index in the slice starting at 8.
            block_index += 8

        block = INDEX_TO_BLOCK[block_index]
        face_ix = position.index(face)
        color = block[rotation + face_ix]

        return color

    def __str__(self) -> str:
        faces = (Face(i) for i in range(6))
        face_colors = {face: self.__get_face_colors(face) for face in faces}
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


BLOCK_TO_INDEX: Dict[Tuple[Color, ...], int] = {
    (Color.WHITE, Color.RED, Color.BLUE): 0,
    (Color.WHITE, Color.BLUE, Color.ORANGE): 1,
    (Color.WHITE, Color.ORANGE, Color.GREEN): 2,
    (Color.WHITE, Color.RED, Color.GREEN): 3,
    (Color.RED, Color.BLUE, Color.YELLOW): 4,
    (Color.BLUE, Color.ORANGE, Color.YELLOW): 5,
    (Color.ORANGE, Color.GREEN, Color.YELLOW): 6,
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

for block in BLOCK_TO_INDEX:
    assert list(block) == sorted(block), block

INDEX_TO_BLOCK = {v: k for k, v in BLOCK_TO_INDEX.items()}

POSITION_TO_INDEX: Dict[Tuple[Face, ...], int] = {
    # Corners:
    (Face.UP, Face.LEFT, Face.FRONT): 0,
    (Face.UP, Face.FRONT, Face.RIGHT): 3,
    (Face.UP, Face.RIGHT, Face.BACK): 6,
    (Face.UP, Face.LEFT, Face.BACK): 9,
    (Face.LEFT, Face.FRONT, Face.DOWN): 12,
    (Face.FRONT, Face.RIGHT, Face.DOWN): 15,
    (Face.RIGHT, Face.BACK, Face.DOWN): 18,
    (Face.LEFT, Face.BACK, Face.DOWN): 21,
    # Edges:
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

for position in POSITION_TO_INDEX:
    assert list(position) == sorted(position), position

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


def main():
    cube = Cube()
    print(cube)


if __name__ == "__main__":
    main()
