import gym
import curses
import logging
import numpy as np
from gym_craftWorld.envs.cookbook import Cookbook
from gym_craftWorld.misc import array
from skimage.measure import block_reduce
from gym import error, spaces, utils
from gym.utils import seeding

WIDTH = 10
HEIGHT = 10

WINDOW_WIDTH = 5
WINDOW_HEIGHT = 5

N_WORKSHOPS = 3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
N_ACTIONS = USE + 1


class craftWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.n_features = \
            2 * WINDOW_WIDTH * WINDOW_HEIGHT * self.cookbook.n_kinds + \
            self.cookbook.n_kinds + \
            4 + \
            1
        self.n_actions = N_ACTIONS

        self.non_grabbable_indices = self.cookbook.environment
        self.grabbable_indices = [i for i in range(self.cookbook.n_kinds)
                                  if i not in self.non_grabbable_indices]
        self.workshop_indices = [self.cookbook.index["workshop%d" % i]
                                 for i in range(N_WORKSHOPS)]
        self.water_index = self.cookbook.index["water"]
        self.stone_index = self.cookbook.index["stone"]

        self.random = np.random.RandomState(0)

        # generate grid
        grid = np.zeros((WIDTH, HEIGHT, self.cookbook.n_kinds))
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[WIDTH - 1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, HEIGHT - 1:, i_bd] = 1

        # treasure
        make_island = False,
        make_cave = False
        if make_island or make_cave:
            (gx, gy) = (1 + np.random.randint(WIDTH - 2), 1)
            treasure_index = \
                self.cookbook.index["gold"] if make_island else self.cookbook.index["gem"]
            wall_index = \
                self.water_index if make_island else self.stone_index
            grid[gx, gy, treasure_index] = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not grid[gx + i, gy + j, :].any():
                        grid[gx + i, gy + j, wall_index] = 1

        # ingredients
        for primitive in self.cookbook.primitives:
            if primitive == self.cookbook.index["gold"] or \
                            primitive == self.cookbook.index["gem"]:
                continue
            for i in range(4):
                (x, y) = random_free(grid, self.random)
                grid[x, y, primitive] = 1

        # generate crafting stations
        for i_ws in range(N_WORKSHOPS):
            ws_x, ws_y = random_free(grid, self.random)
            grid[ws_x, ws_y, self.cookbook.index["workshop%d" % i_ws]] = 1

        # generate init pos
        init_pos = random_free(grid, self.random)

        return CraftScenario(grid, init_pos, self)


    def step(self, action):

    def reset(self):

    def render(self, mode='human', close=False):

def random_free(grid, random):
    pos = None
    while pos is None:
        (x, y) = (random.randint(WIDTH), random.randint(HEIGHT))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos

def neighbors(pos, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == LEFT):
        neighbors.append((x - 1, y))
    if y > 0 and (dir is None or dir == DOWN):
        neighbors.append((x, y - 1))
    if x < WIDTH - 1 and (dir is None or dir == RIGHT):
        neighbors.append((x + 1, y))
    if y < HEIGHT - 1 and (dir is None or dir == UP):
        neighbors.append((x, y + 1))
    return neighbors


class CraftState(object):

    def satisfies(self, goal_name, goal_arg):
        return self.inventory[goal_arg] > 0

    def features(self):
        if self._cached_features is None:
            x, y = self.pos
            hw = WINDOW_WIDTH / 2
            hh = WINDOW_HEIGHT / 2
            bhw = (WINDOW_WIDTH * WINDOW_WIDTH) / 2
            bhh = (WINDOW_HEIGHT * WINDOW_HEIGHT) / 2

            grid_feats = array.pad_slice(self.grid, (x - hw, x + hw + 1),
                                         (y - hh, y + hh + 1))
            grid_feats_big = array.pad_slice(self.grid, (x - bhw, x + bhw + 1),
                                             (y - bhh, y + bhh + 1))
            grid_feats_big_red = block_reduce(grid_feats_big,
                                              (WINDOW_WIDTH, WINDOW_HEIGHT, 1), func=np.max)
            # grid_feats_big_red = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, self.world.cookbook.n_kinds))

            self.gf = grid_feats.transpose((2, 0, 1))
            self.gfb = grid_feats_big_red.transpose((2, 0, 1))

            pos_feats = np.asarray(self.pos)
            pos_feats[0] /= WIDTH
            pos_feats[1] /= HEIGHT

            dir_features = np.zeros(4)
            dir_features[self.dir] = 1

            features = np.concatenate((grid_feats.ravel(),
                                       grid_feats_big_red.ravel(), self.inventory,
                                       dir_features, [0]))
            assert len(features) == self.world.n_features
            self._cached_features = features

        return self._cached_features


    def next_to(self, i_kind):
        x, y = self.pos
        return self.grid[x - 1:x + 2, y - 1:y + 2, i_kind].any()

class CraftScenario(object):
    def __init__(self, grid, init_pos, world):
        self.init_grid = grid
        self.init_pos = init_pos
        self.init_dir = 0
        self.world = world

    def init(self):
        inventory = np.zeros(self.world.cookbook.n_kinds)
        state = CraftState(self, self.init_grid, self.init_pos, self.init_dir, inventory)
        return state