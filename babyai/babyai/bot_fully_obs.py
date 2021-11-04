from gym_minigrid.minigrid import *
from babyai.levels.verifier import *
from babyai.levels.verifier import (ObjDesc, pos_next_to,
                                    GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, AndInstr, AfterInstr)
from collections import namedtuple
from babyai.bot import Subgoal, CloseSubgoal, OpenSubgoal, DropSubgoal, PickupSubgoal, GoNextToSubgoal, ExploreSubgoal, LanguageObj
import numpy as np

class DisappearedBoxError(Exception):
    """
    Error that's thrown when a box is opened.
    We make the assumption that the bot cannot accomplish the mission when it happens.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def manhattan_distance(pos, target):
    return np.abs(target[0] - pos[0]) + np.abs(target[1] - pos[1])

class FullyObsBot:
    """A bot that can solve all BabyAI levels.

    The bot maintains a plan, represented as a stack of the so-called
    subgoals. The initial set of subgoals is generated from the instruction.
    The subgoals are then executed one after another, unless a change of
    plan is required (e.g. the location of the target object is not known
    or there other objects in the way). In this case, the bot changes the plan.

    The bot can also be used to advice a suboptimal agent, e.g. play the
    role of an oracle in algorithms like DAGGER. It changes the plan based on
    the actual action that the agent took.

    The main method of the bot (and the only one you are supposed to use) is `replan`.

    Parameters:
    ----------
    mission : a freshly created BabyAI environment

    """

    def __init__(self, mission):
        # Mission to be solved
        self.mission = mission

        self.subgoal_counter = 0

        # Grid containing what has been mapped out
        self.grid = Grid(mission.width, mission.height)

        # Visibility mask. True for explored/seen, false for unexplored.
        self.vis_mask = np.zeros(shape=(mission.width, mission.height), dtype=np.bool)

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions
        self._process_instr(mission.instrs)

        # How many BFS searches this bot has performed
        self.bfs_counter = 0

        # How many steps were made in total in all BFS searches
        # performed by this bot
        self.bfs_step_counter = 0


    def replan(self, action_taken=None, give_subgoal=False):
        """Replan and suggest an action.

        Call this method once per every iteration of the environment.

        Parameters:
        ----------
        action_taken
            The last action that the agent took. Can be `None`,
            in which case the bot assumes that the action it suggested
            was taken (or that it is the first iteration).

        Returns:
        -------
        suggested_action
            The action that the bot suggests. Can be `done` if the
            bot thinks that the mission has been accomplished.

        """
        self._process_obs()

        # Check that no box has been opened
        self._check_erroneous_box_opening(action_taken)

        # TODO: instead of updating all subgoals, just add a couple
        # properties to the `Subgoal` class.
        for subgoal in self.stack:
            subgoal.update_agent_attributes()

        if self.stack:
            self.stack[-1].replan_after_action(action_taken)

        # Clear the stack from the non-essential subgoals
        while self.stack and self.stack[-1].is_exploratory():
            self.stack.pop()

        suggested_action = None
        while self.stack:
            subgoal = self.stack[-1]
            suggested_action = subgoal.replan_before_action()
            # If is not clear what can be done for the current subgoal
            # (because it is completed, because there is blocker,
            # or because exploration is required), keep replanning
            if suggested_action is not None:
                break
        if not self.stack:
            suggested_action = self.mission.actions.done

        self._remember_current_state()
        if give_subgoal:
            return suggested_action, subgoal
        return suggested_action

    def _find_obj_pos(self, obj_desc, adjacent=False):
        """Find the position of the closest visible object matching a given description."""

        assert len(obj_desc.obj_set) > 0

        best_distance_to_obj = 999
        best_pos = None
        best_obj = None

        for i in range(len(obj_desc.obj_set)):
            try:
                if obj_desc.obj_set[i] == self.mission.carrying:
                    continue
                obj_pos = obj_desc.obj_poss[i]

                if self.vis_mask[obj_pos]:
                    shortest_path_to_obj, _, with_blockers, with_unlock = self._shortest_path(
                        lambda pos, cell: pos == obj_pos,
                        try_with_blockers=True, ignore_doors=True, try_wtih_unlock=True
                    )
                    assert shortest_path_to_obj is not None
                    distance_to_obj = len(shortest_path_to_obj)

                    if with_blockers:
                        # The distance should take into account the steps necessary
                        # to unblock the way. Instead of computing it exactly,
                        # we can use a lower bound on this number of steps
                        # which is 4 when the agent is not holding anything
                        # (pick, turn, drop, turn back
                        # and 7 if the agent is carrying something
                        # (turn, drop, turn back, pick,
                        # turn to other direction, drop, turn back)
                        distance_to_obj = (len(shortest_path_to_obj)
                                           + (7 if self.mission.carrying else 4))

                    if with_unlock:
                        # The maximum distance to get a key is assumed to be traveling width + height twice
                        distance_to_obj = distance_to_obj + 2*(self.mission.width + self.mission.height)

                    # If we looking for a door and we are currently in that cell
                    # that contains the door, it will take us at least 2
                    # (3 if `adjacent == True`) steps to reach the goal.`
                    if distance_to_obj == 0:
                        distance_to_obj = 3 if adjacent else 2

                    # If what we want is to face a location that is adjacent to an object,
                    # and if we are already right next to this object,
                    # then we should not prefer this object to those at distance 2
                    if adjacent and distance_to_obj == 1:
                        distance_to_obj = 3

                    if distance_to_obj < best_distance_to_obj:
                        best_distance_to_obj = distance_to_obj
                        best_pos = obj_pos
                        best_obj = obj_desc.obj_set[i]
            except IndexError:
                # Suppose we are tracking red keys, and we just used a red key to open a door,
                # then for the last i, accessing obj_desc.obj_poss[i] will raise an IndexError
                # -> Solution: Not care about that red key we used to open the door
                pass

        return best_obj, best_pos

    def _process_obs(self):
        """Parse the contents of an observation/image and update our state."""

        grid, vis_mask = self.mission.gen_obs_grid()

        view_size = self.mission.agent_view_size
        pos = self.mission.agent_pos
        f_vec = self.mission.dir_vec
        r_vec = self.mission.right_vec

        # Compute the absolute coordinates of the top-left corner
        # of the agent's view area
        top_left = pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)

        # Mark everything in front of us as visible
        for vis_j in range(0, view_size):
            for vis_i in range(0, view_size):

                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.vis_mask.shape[0]:
                    continue
                if abs_j < 0 or abs_j >= self.vis_mask.shape[1]:
                    continue

                self.vis_mask[abs_i, abs_j] = True

    def _remember_current_state(self):
        self.prev_agent_pos = self.mission.agent_pos
        self.prev_carrying = self.mission.carrying
        fwd_cell = self.mission.grid.get(*self.mission.agent_pos + self.mission.dir_vec)
        if fwd_cell and fwd_cell.type == 'door':
            self.fwd_door_was_open = fwd_cell.is_open
        self.prev_fwd_cell = fwd_cell

    def _closest_wall_or_door_given_dir(self, position, direction):
        distance = 1
        while True:
            position_to_try = position + distance * direction
            # If the current position is outside the field of view,
            # stop everything and return the previous one
            if not self.mission.in_view(*position_to_try):
                return distance - 1
            cell = self.mission.grid.get(*position_to_try)
            if cell and (cell.type.endswith('door') or cell.type == 'wall'):
                return distance
            distance += 1

    def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers, ignore_doors=False, ignore_locks=False):
        """Performs breadth first search.

        This is pretty much your textbook BFS. The state space is agent's locations,
        but the current direction is also added to the queue to slightly prioritize
        going straight over turning.

        """
        self.bfs_counter += 1

        queue = [(state, None) for state in initial_states]
        grid = self.mission.grid
        previous_pos = dict()

        while len(queue) > 0:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j, di, dj = state

            if (i, j) in previous_pos:
                continue

            self.bfs_step_counter += 1

            cell = grid.get(i, j)
            previous_pos[(i, j)] = prev_pos

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), cell):
                path = []
                pos = (i, j)
                while pos:
                    path.append(pos)
                    pos = previous_pos[pos]
                return path, (i, j), previous_pos

            # If this cell was not visually observed, don't expand from it
            if not self.vis_mask[i, j]:
                continue

            if cell:
                if cell.type == 'wall':
                    continue
                # If this is a door
                elif cell.type == 'door':
                    # If the door is closed, don't visit neighbors
                    if cell.is_locked and not ignore_locks:
                        continue
                    if not cell.is_open and not ignore_doors and not ignore_blockers:
                        continue
                elif not ignore_blockers:
                    continue

            # Location to which the bot can get without turning
            # are put in the queue first
            for k, l in [(di, dj), (dj, di), (-dj, -di), (-di, -dj)]:
                next_pos = (i + k, j + l)
                next_dir_vec = (k, l)
                next_state = (*next_pos, *next_dir_vec)
                queue.append((next_state, (i, j)))

        # Path not found
        return None, None, previous_pos

    def _shortest_path(self, accept_fn, try_with_blockers=False, ignore_doors=False, try_wtih_unlock=False):
        """
        Finds the path to any of the locations that satisfy `accept_fn`.
        Prefers the paths that avoid blockers for as long as possible.
        """

        # Initial states to visit (BFS)
        initial_states = [(*self.mission.agent_pos, *self.mission.dir_vec)]

        path = finish = None
        with_blockers = False
        with_unlocks = False
        path, finish, previous_pos = self._breadth_first_search(
            initial_states, accept_fn, ignore_blockers=False, ignore_doors=ignore_doors)
        if not path and try_with_blockers:
            with_blockers = True
            path, finish, _ = self._breadth_first_search(
                [(i, j, 1, 0) for i, j in previous_pos],
                accept_fn, ignore_blockers=True, ignore_doors=ignore_doors)
            if path:
                # `path` now contains the path to a cell that is reachable without
                # blockers. Now let's add the path to this cell
                pos = path[-1]
                extra_path = []
                while pos:
                    extra_path.append(pos)
                    pos = previous_pos[pos]
                path = path + extra_path[1:]

        if not path and try_wtih_unlock:
            with_unlocks = True
            path, finish, _ = self._breadth_first_search(
                [(i, j, 1, 0) for i, j in previous_pos],
                accept_fn, ignore_blockers=True, ignore_doors=ignore_doors, ignore_locks=True)
            if path:
                # `path` now contains the path to a cell that is reachable without
                # blockers. Now let's add the path to this cell
                pos = path[-1]
                extra_path = []
                while pos:
                    extra_path.append(pos)
                    pos = previous_pos[pos]
                path = path + extra_path[1:]

        if path:
            # And the starting position is not required
            path = path[::-1]
            path = path[1:]

        # Note, that with_blockers only makes sense if path is not None
        if not try_wtih_unlock:
            return path, finish, with_blockers
        else:
            return path, finish, with_blockers, with_unlocks

    def _find_drop_pos(self, except_pos=None):
        """
        Find a position where an object can be dropped, ideally without blocking anything.
        """

        grid = self.mission.grid

        def match_unblock(pos, cell):
            # Consider the region of 8 neighboring cells around the candidate cell.
            # If dropping the object in the candidate makes this region disconnected,
            # then probably it is better to drop elsewhere.

            i, j = pos
            agent_pos = tuple(self.mission.agent_pos)

            if np.array_equal(pos, agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[i, j] or grid.get(i, j):
                return False

            # We distinguish cells of three classes:
            # class 0: the empty ones, including open doors
            # class 1: those that are not interesting (just walls so far)
            # class 2: all the rest, including objects and cells that are current not visible,
            #          and hence may contain objects, and also `except_pos` at it may soon contain
            #          an object
            # We want to ensure that empty cells are connected, and that one can reach
            # any object cell from any other object cell.
            cell_class = []
            for k, l in [(-1, -1), (0, -1), (1, -1), (1, 0),
                         (1, 1), (0, 1), (-1, 1), (-1, 0)]:
                nb_pos = (i + k, j + l)
                cell = grid.get(*nb_pos)
                # compeletely blocked
                if self.vis_mask[nb_pos] and cell and cell.type == 'wall':
                    cell_class.append(1)
                # empty
                elif (self.vis_mask[nb_pos]
                        and (not cell or (cell.type == 'door' and cell.is_open) or nb_pos == agent_pos)
                        and nb_pos != except_pos):
                    cell_class.append(0)
                # an object cell
                else:
                    cell_class.append(2)

            # Now we need to check that empty cells are connected. To do that,
            # let's check how many times empty changes to non-empty
            changes = 0
            for i in range(8):
                if bool(cell_class[(i + 1) % 8]) != bool(cell_class[i]):
                    changes += 1

            # Lastly, we need check that every object has an adjacent empty cell
            for i in range(8):
                next_i = (i + 1) % 8
                prev_i = (i + 7) % 8
                if cell_class[i] == 2 and cell_class[prev_i] != 0 and cell_class[next_i] != 0:
                    return False

            return changes <= 2

        def match_empty(pos, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[pos] or grid.get(*pos):
                return False

            return True

        _, drop_pos, _ = self._shortest_path(match_unblock)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_unblock, try_with_blockers=True)

        if not drop_pos:
            _, drop_pos, _ = self._shortest_path(match_empty, try_with_blockers=True)

        return drop_pos

    def _process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(OpenSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc, reason='Open'))
            return

        if isinstance(instr, PickupInstr):
            # We pick up and immediately drop so
            # that we may carry other objects
            self.stack.append(DropSubgoal(self))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(DropSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_fixed, reason='PutNext'))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_move))
            return

        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr):
            self._process_instr(instr.instr_b)
            self._process_instr(instr.instr_a)
            return

        if isinstance(instr, AfterInstr):
            self._process_instr(instr.instr_a)
            self._process_instr(instr.instr_b)
            return

        assert False, "unknown instruction type"

    def _check_erroneous_box_opening(self, action):
        """
        When the agent opens a box, we raise an error and mark the task unsolvable.
        This is a tad conservative, because maybe the box is irrelevant to the mission.
        """
        if (action == self.mission.actions.toggle
                and self.prev_fwd_cell is not None
                and self.prev_fwd_cell.type == 'box'):
            raise DisappearedBoxError('A box was opened. I am not sure I can help now.')
