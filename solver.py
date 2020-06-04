'''Solving the cube given a value model.'''

import abc
import collections
import dataclasses
import heapq

from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

import cube as cube_lib
import util


@dataclasses.dataclass
class _Trajectory:
    final_state: cube_lib.Cube
    first_rotation: Optional[cube_lib.Rotation] = None
    num_rotations: int = 0


class Solver(metaclass=abc.ABCMeta):
    '''An object that can solve a Rubik's cube.

    It is initialized with a cube, and `apply_next_rotation` can be called
    repeatedly to solve the cube one rotation at a time.'''
    def __init__(self, cube: cube_lib.Cube):
        self._cube: cube_lib.Cube = cube.copy()

    @abc.abstractmethod
    def apply_next_rotation(self) -> cube_lib.Rotation:
        '''Applies one rotation to the cube.

        Returns the rotation that was applied, or None if the cube was already
        solved.
        '''

    @property
    def cube(self) -> cube_lib.Cube:
        '''Returns the cube being solved.

        `apply_next_rotation` modifies this same cube, so callers shouldn't
        modify it.'''
        return self._cube


class GreedySolver(Solver):
    '''A solver that uses a greedy algorithm.

    It looks ahead in the tree of future states up to a given depth, evaluates
    states using the provided model, and picks the next rotation that gets us
    closer to solving the cube.
    '''
    def __init__(self, cube: cube_lib.Cube, model: tf.keras.Model, depth: int):
        '''
        The model is assumed to predict the distance towards the solved state,
        i.e. the number of rotations that will need to be applied in order to
        solve the cube (this is the opposive of a value function).
        '''
        super().__init__(cube)
        self._model = model
        self._depth = depth

    def apply_next_rotation(self):
        if self.cube.is_solved():
            return None

        model_batcher = util.ModelBatcher(1024,
                                          self._model,
                                          feature_shape=(20, 24))
        # We do a BFS so that we traverse states in increasing order of depth.
        # That way, as soon as we encounter a solved state, we know that we
        # have found the shortest path.
        queue = collections.deque()
        queue.append(_Trajectory(final_state=self.cube))

        explored_set = {self.cube}

        best_rotation = None
        best_value = None

        def process_predictions():
            nonlocal best_rotation
            nonlocal best_value
            for value, first_rotation in model_batcher.get_predictions():
                value = value[0]
                # The model predicts the distance, so we minimize it.
                if best_value is None or value < best_value:
                    best_value = value
                    best_rotation = first_rotation

        while queue:
            trajectory = queue.pop()
            state = trajectory.final_state

            if state.is_solved():
                # We know this is the shortest trajectory since we are doing a
                # BFS.
                self.cube.rotate_face(trajectory.first_rotation)
                return trajectory.first_rotation

            if trajectory.num_rotations >= self._depth:
                # We reached a leaf in the tree, therefore we use the model to
                # evaluate the state.
                model_batcher.enqueue_predictions(
                    state.as_numpy_array().reshape((1, 20, 24)),
                    request_id=trajectory.first_rotation)

                process_predictions()
            else:
                # This isn't a leaf state, we expand it.
                for rotation in cube_lib.Rotation.all():
                    new_state = state.copy()
                    new_state.rotate_face(rotation)

                    if new_state not in explored_set:
                        explored_set.add(new_state)
                        # The first_rotation is set to None for empty
                        # trajectory (the one with the root state). If that's
                        # the case, then we are currently at the first
                        # rotation.
                        first_rotation = trajectory.first_rotation or rotation
                        queue.appendleft(
                            _Trajectory(
                                final_state=new_state,
                                first_rotation=first_rotation,
                                num_rotations=trajectory.num_rotations + 1))

        model_batcher.flush()
        process_predictions()

        self.cube.rotate_face(best_rotation)
        return best_rotation


@dataclasses.dataclass
class _AStarState:
    '''An entry in the priority queue of the AStarSolver.'''
    cube: cube_lib.Cube
    cost_to_come: float
    est_cost_to_go: float
    previous_state: Optional[cube_lib.Cube]
    previous_rotation: Optional[cube_lib.Rotation]

    # Used internally by the PriotityQueue, to denote that an entry in the queue
    # has been invalidated by an update.
    is_valid: bool = True

    # TODO: implement comparison operators.


class _PriorityQueue:
    def __init__(self):
        # The tuple contains:
        #  - the value of the state.
        #  - an arbitrary (but unique) int, to break ties, because _AStartState
        #    doesn't support comparisons.
        self._list: List[Tuple[float, int, _AStarState]] = []
        # To allow for updating the value of a state, we keep a dict that points
        # to the current valid entry in self._list for a given cube. When its
        # value is updated, this entry is marked as invalid.
        self._dict: Dict[cube_lib.Cube, _AStarState] = {}
        self._num_insertions: int = 0

    def pop_min_state(self) -> _AStarState:
        '''Returns the state with the lowest value.  '''

        while self._list:
            _, _, state = heapq.heappop(self._list)
            if state.is_valid:
                del self._dict[state.cube]
                return state
        raise IndexError('Empty PriorityQueue')

    def add_or_update_state(self, state: _AStarState) -> None:
        '''Adds a state to the priority queue.

        If the same state is already present in the queue (i.e. a state that
        represents the same cube configuration), then update its value to the
        minimum of the previous value and the new value.
        '''
        existing_state = self._dict.get(state.cube)
        if existing_state:
            if self._value(existing_state) < self._value(state):
                # The same state already exists with a lower value, so we can
                # ignore the update.
                return
            # We need to update the value of the state already present in the
            # queue: we invalidate the existing entry, and we will add a new
            # one.
            existing_state.is_valid = False

        value = self._value(state)
        heapq.heappush(self._list, (value, self._num_insertions, state))
        self._num_insertions += 1
        self._dict[state.cube] = state

    def __bool__(self) -> bool:
        for _, _, state in self._list:
            if state.is_valid:
                return True
        return False

    @staticmethod
    def _value(state: _AStarState) -> float:
        return state.cost_to_come + state.est_cost_to_go


class AStarSolver(Solver):
    '''A solver that uses A* to find a solution. '''
    def __init__(self, cube: cube_lib.Cube, model: tf.keras.Model):
        super().__init__(cube)
        self._model: tf.keras.Model = model
        self._solution: Optional[Deque[cube_lib.Rotation]] = None

    def apply_next_rotation(self):
        rotation = self._get_next_rotation()
        if rotation is None:
            return None
        self.cube.rotate_face(rotation)
        return rotation

    def _get_next_rotation(self) -> Optional[cube_lib.Rotation]:
        if self._solution:
            return self._solution.popleft()
        if self._solution is not None:
            # We are done applying the rotations in the solution.
            assert self.cube.is_solved
            return None

        if self.cube.is_solved():
            return None

        done_states: Dict[cube_lib.Cube, _AStarState] = {}
        queue = _PriorityQueue()

        est_distance = self._model.predict(self.cube.as_numpy_array().reshape(
            (1, 20, 24))).item()
        queue.add_or_update_state(
            _AStarState(cube=self.cube,
                        cost_to_come=0,
                        est_cost_to_go=est_distance,
                        previous_state=None,
                        previous_rotation=None))

        while queue:
            state = queue.pop_min_state()
            if state.cube.is_solved():
                self._solution = self._compute_solution(state, done_states)
                # The solution is cached, we can just recurse and it will read
                # the first move from the cached solution.
                return self._get_next_rotation()

            model_batcher = util.ModelBatcher(cube_lib.NUM_ROTATIONS,
                                              self._model,
                                              feature_shape=(20, 24))

            for rotation in cube_lib.Rotation.all():
                new_cube = state.cube.copy()
                new_cube.rotate_face(rotation)

                new_state = _AStarState(
                    cube=new_cube,
                    cost_to_come=state.cost_to_come + 1,
                    est_cost_to_go=-1,  # will be set with the model prediction.
                    previous_state=state.cube,
                    previous_rotation=rotation)

                model_batcher.enqueue_predictions(
                    new_cube.as_numpy_array().reshape((1, 20, 24)),
                    request_id=new_state)

            model_batcher.flush()
            for value, new_state in model_batcher.get_predictions():
                new_state.est_cost_to_go = value
                queue.add_or_update_state(new_state)

            done_states[state.cube] = state

        assert False  # This could should be unreachable.

    @staticmethod
    def _compute_solution(
        goal_state: _AStarState,
        done_states: Dict[cube_lib.Cube,
                          _AStarState]) -> Deque[cube_lib.Rotation]:
        '''Computes the full solution (list of rotations) to solve the cube. '''
        solution: Deque[cube_lib.Rotation] = collections.deque()
        state: _AStarState = goal_state
        while state.previous_rotation:
            assert state.previous_state is not None
            solution.appendleft(state.previous_rotation)
            state = done_states[state.previous_state]
        return solution


def num_steps_to_solve(solver: Solver, max_num_steps: int):
    '''Returns the number of steps required to solve a cube.

    Tries `max_num_steps` steps, and returns None if the cube is not solved
    after that.
    '''
    for step in range(max_num_steps):
        if solver.cube.is_solved():
            return step
        solver.apply_next_rotation()

    if solver.cube.is_solved():
        return max_num_steps

    return None


def evaluate_solver(solver_fn: Callable[[cube_lib.Cube],
                                        Solver], trajectory_length: int,
                    max_num_steps: int, num_trials: int):
    '''Evaluates a solver.

    Uses the solver to solves `num_trials` cubes that are scrambled with
    `trajectory_length` random rotations.

    Returns a dataframe with statistics about the performance of the solver.
    '''
    evaluation_results = pd.DataFrame()
    for _ in range(num_trials):
        cube = cube_lib.get_scrambled_cube(trajectory_length)
        solver = solver_fn(cube)

        num_steps = num_steps_to_solve(solver, max_num_steps)

        evaluation_results = evaluation_results.append(
            {
                'num_steps_scrambled': trajectory_length,
                'num_steps_to_solve': num_steps,
                'solved': solver.cube.is_solved()
            },
            ignore_index=True)
    evaluation_results.num_steps_scrambled = (
        evaluation_results.num_steps_scrambled.astype('int'))
    # We can't convert 'num_steps_to_solve' to ints, because it contains NAs,
    # which are of type float.
    evaluation_results.num_steps_to_solve = (
        evaluation_results.num_steps_to_solve.astype('float'))
    evaluation_results.solved = evaluation_results.solved.astype('bool')
    return evaluation_results


def main():
    model = tf.keras.models.load_model('./model')
    evaluation = evaluate_solver(lambda cube: GreedySolver(cube, model, 2),
                                 trajectory_length=10,
                                 max_num_steps=15,
                                 num_trials=20)
    print(evaluation)


if __name__ == '__main__':
    main()
