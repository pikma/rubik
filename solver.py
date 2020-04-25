import collections

import tensorflow as tf

import cube as cube_lib

Trajectory = collections.namedtuple('Trajectory', ['final_state', 'rotations'])


def greedy_solve(model: tf.keras.Model,
                 cube: cube_lib.Cube,
                 depth: int = 1) -> cube_lib.Rotation:
    '''Returns the next rotation.

    Performs a greedy search of the tree of states, bounded to a depth.
    Evaluates states using the provided model.

    Returns None if the cube is already solved.
    '''

    if cube.is_solved():
        return None

    # We do a BFS so that we traverse states in increasing order of depth. That
    # way, as soon as we encounter a solved state, we know that we have found
    # the shortest path.
    queue = collections.deque([Trajectory(final_state=cube, rotations=[])])
    explored_set = {cube}

    best_rotation = None
    best_value = None

    while queue:
        trajectory = queue.pop()
        state = trajectory.final_state

        if state.is_solved():
            # We know this is the shortest trajectory since we are doing a BFS.
            return trajectory.rotations[0]

        if len(trajectory.rotations) >= depth:
            # Evaluate the state. The model predicts the distance to a solved
            # state, so the value is the opposite.
            value = -model.predict([[state.as_numpy_array()]])[0]
            if best_value is None or value > best_value:
                best_value = value
                best_rotation = trajectory.rotations[0]
            continue

        for rotation in cube_lib.Rotation.all():
            new_state = state.copy()
            new_state.rotate_face(rotation)

            if new_state not in explored_set:
                explored_set.add(new_state)
                queue.appendleft(
                    Trajectory(final_state=new_state,
                               rotations=trajectory.rotations + [rotation]))

    return best_rotation


