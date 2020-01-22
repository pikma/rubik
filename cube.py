import enum

import numpy as np

@enum.unique
class Color(enum.Enum):
  WHITE = 0
  RED = 1
  BLUE = 2
  ORANGE = 3
  GREEN = 4
  YELLOW = 5

@enum.unique
class Face(enum.Enum):
  '''A face of the Cube. This is the part that can be rotated.'''
  U = 0
  L = 1
  F = 2
  R = 3
  B = 4
  D = 5


class Cube:
  ''' A Rubik's cube.

  It is represented internally as a one-hot 20 * 24 matrix. Each row represents
  a block (corner or edge), and for each row, exactly one column is set to 1.
  There are 24 possible positions for each block (for angles, 8 positions * 3
  orientations; for edges, 12 positions * 2 orientations), therefore the column
  that is set to 1 represents the position and orientation of the block.

  This idea comes from McAleer 2018 (https://arxiv.org/abs/1805.07470).
  '''

  def __init__(self):
    # Corners:
    #   0: White Red Blue
    #   1: White Blue Orange
    #   2: White Orange Green
    #   3: White Green Red
    #   4: Red Yellow Blue
    #   5: Blue Yellow Orange
    #   6: Orange Yellow Green
    #   7: Green Yellow Red
    #
    # Angles:
    #   8: White Red
    #   9: White Blue
    #   10: White Orange
    #   11: White Green
    #   12: Red Blue
    #   13: Blue Orange
    #   14: Orange Green
    #   15: Green Red
    #   16: Blue Yellow
    #   17: Orange Yellow
    #   18: Green Yellow
    #   19: Red Yellow
    #
    # Positions for corner:
    #   Each position is determined by the faces that the block is on.
    #
    #   For each entry in the list below, there are three possible orientations
    #   of the block. To encode the orientation, we look at where the sticker
    #   with the first color (in the order of the Color enum) is. If it is on
    #   the first face in the list given below (for "Up Left Front", Up), then
    #   the index in the array is 0 mod 3. If it is on the second face (Left),
    #   then it is 1 mod 3, and if it is on the third face (Front), then it is 2
    #   mod 3.
    #
    #   0-2: Up Left Front
    #   3-5: Up Front Right
    #   6-8: Up Right Back
    #   9-11: Up Back Left
    #   12-14: Left Down Front
    #   15-17: Front Down Right
    #   18-20: Right Down Back
    #   21-23: Back Down Left
    #
    # Positions for edge:
    #   For each, the position is 0 mod 2 if the edge's first color (in the
    #   order of the Color enum) is on the first listed below, and 1 mod 2 if it
    #   is on the second face.
    #   0-1: Up Left
    #   2-3: Up Front
    #   4-5: Up Right
    #   6-7: Up Back
    #   8-9: Left Front
    #   10-11: Front Right
    #   12-13: Right Back
    #   14-15: Left Back
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


    def __str__(self):
      # TODO: turn the giant comments above into code that this method can read
      # to programatically find the color on each face.
      pass

