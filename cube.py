import array
import enum

@enum.unique
class Color(enum.Enum):
  WHITE = 0
  BLUE = 1
  RED = 2
  GREEN = 3
  ORANGE = 4
  YELLOW = 5


class Cube:
  ''' A Rubik's cube.'''
  # To keep track of the cube's state, we always assume it is held such that
  # white is on top, blue in front, red on the left, orange on the right, green
  # in the back, and yellow on the bottom.
  #
  def __init__(self):
    # We represent the cube in a contiguous array of blocks, each block is
    # represented by an integer internally. The goal is to make state
    # manipulations as fast and cheap as possible.
    #
    # Blocks are stored level by level (top to bottom), each level starting on
    # the blue/red corner, and then continuing counter-clockwise.
    self.blocks = array.array('I',
        [
          # Top level (white):
          Cube._uint_from_corner(Color.WHITE, Color.RED, Color.BLUE),
          Cube._uint_from_angle(Color.WHITE, Color.BLUE),
          Cube._uint_from_corner(Color.WHITE, Color.BLUE, Color.ORANGE),
          Cube._uint_from_angle(Color.WHITE, Color.ORANGE),
          Cube._uint_from_corner(Color.WHITE, Color.ORANGE, Color.GREEN),
          Cube._uint_from_angle(Color.WHITE, Color.GREEN),
          Cube._uint_from_corner(Color.WHITE, Color.GREEN, Color.RED),
          Cube._uint_from_angle(Color.WHITE, Color.RED),

          # Middle level:
          Cube._uint_from_angle(Color.RED, Color.BLUE),
          Cube._uint_from_angle(Color.BLUE, Color.ORANGE),
          Cube._uint_from_angle(Color.ORANGE, Color.GREEN),
          Cube._uint_from_angle(Color.GREEN, Color.RED),

          # Bottom level (yellow):
          Cube._uint_from_corner(Color.RED, Color.BLUE, Color.YELLOW),
          Cube._uint_from_angle(Color.BLUE, Color.YELLOW),
          Cube._uint_from_corner(Color.BLUE, Color.ORANGE, Color.YELLOW),
          Cube._uint_from_angle(Color.ORANGE, Color.YELLOW),
          Cube._uint_from_corner(Color.ORANGE, Color.GREEN, Color.YELLOW),
          Cube._uint_from_angle(Color.GREEN, Color.YELLOW),
          Cube._uint_from_corner(Color.GREEN, Color.RED, Color.YELLOW),
          Cube._uint_from_angle(Color.RED, Color.YELLOW) ])

  # To represent blocks as uints, we use the [0, 35] range for angles, and the
  # [36, 251] range for blocks.

  @staticmethod
  def _uint_from_corner(first_color, second_color, third_color):
    '''
    The colors should be listed in the following order:
      top, then counter-clockwise, then bottom.
    '''
    return 36 + first_color.value * 36 + second_color.value * 6 + third_color.value


  @staticmethod
  def _uint_from_angle(first_color, second_color):
    '''
    The colors should be listed in the following order:
      top, left, front, right, bottom.
    '''
    return first_color.value * 6 + second_color.value

  @staticmethod
  def _first_color_from_angle_uint(angle_uint):
    return Color(angle_uint // 6)

  @staticmethod
  def _second_color_from_angle_uint(angle_uint):
    return Color(angle_uint % 6)

  @staticmethod
  def _first_color_from_corner_uint(angle_uint):
    return Color((angle_uint - 36) // 36)

  @staticmethod
  def _second_color_from_corner_uint(angle_uint):
    return Color(((angle_uint - 36) // 6) % 6)

  @staticmethod
  def _third_color_from_corner_uint(angle_uint):
    return Color((angle_uint - 36)  % 6)


  def __str__(self):
    colors = [
    # Top face:
    Cube._first_color_from_corner_uint(self.blocks[6]),
    Cube._first_color_from_angle_uint(self.blocks[5]),
    Cube._first_color_from_corner_uint(self.blocks[4]),
    Cube._first_color_from_angle_uint(self.blocks[7]),
    Color.WHITE,
    Cube._first_color_from_angle_uint(self.blocks[3]),
    Cube._first_color_from_corner_uint(self.blocks[0]),
    Cube._first_color_from_angle_uint(self.blocks[1]),
    Cube._first_color_from_corner_uint(self.blocks[2]),

    # Front face:
    Cube._third_color_from_corner_uint(self.blocks[0]),
    Cube._second_color_from_angle_uint(self.blocks[1]),
    Cube._second_color_from_corner_uint(self.blocks[2]),
    Cube._second_color_from_angle_uint(self.blocks[8]),
    Color.BLUE,
    Cube._first_color_from_angle_uint(self.blocks[9]),
    Cube._second_color_from_corner_uint(self.blocks[12]),
    Cube._first_color_from_angle_uint(self.blocks[13]),
    Cube._first_color_from_corner_uint(self.blocks[14]),

    # Left face:
    Cube._third_color_from_corner_uint(self.blocks[6]),
    Cube._second_color_from_angle_uint(self.blocks[7]),
    Cube._second_color_from_corner_uint(self.blocks[0]),
    Cube._second_color_from_angle_uint(self.blocks[11]),
    Color.RED,
    Cube._first_color_from_angle_uint(self.blocks[8]),
    Cube._second_color_from_corner_uint(self.blocks[18]),
    Cube._first_color_from_angle_uint(self.blocks[19]),
    Cube._first_color_from_corner_uint(self.blocks[12]),
    ]

    for i in range(len(colors)//3):
      print('{} {} {}'.format(colors[3*i], colors[3*i+1], colors[3*i + 2]))



