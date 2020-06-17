import gym
from gym import spaces
import numpy as np
from enum import Enum
import numpy as np

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    UP = 4
    DOWN = 5

class Drone(object):
    def __init__(self, pos, fov):
        self.fov = np.radians(fov)
        self.pos = pos
        self.view = None

class FOIEnv(gym.Env):
    """Field of interest environment for drone coverage problem."""

    metadata = {'render.modes': ['human']}
    
    def __init__(self, env_shape, foi=None, fov=30, n_drones=1, max_steps=2000, seed=None):
        super().__init__()
        assert len(env_shape) == 3, 'Environment shape must be of form (X, Y, Z).'
        self.action_space = spaces.Discrete(6)
        self.env_shape = env_shape
        self.observation_space = spaces.Box(low=0, high=np.array(env_shape))
        self.fov = fov
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.foi = foi
        
        # self._gen_map(seed=seed)
        self._drones = {}
        # self._init_drones()
        self._current_steps = 0
        self._done = False

    def step(self, action):
        assert len(action) == len(self._drones), 'Must include actions for every drone.'
        for drone, a in action.items():
            self._drone_action(drone, Action(a))
        observation = self._state()
        reward = self._gr()
        done = self._done or self._current_steps == self.max_steps
        return observation, reward, done, {'success': self._done}

    def reset(self, map_seed=None, drone_seed=None):
        self._gen_map(seed=map_seed)
        self._init_drones(seed=drone_seed)
        observation = self._state()
        self._current_steps = 0
        self._done = False
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _gen_map(self, seed=None):
        X_env, Y_env, Z_env = self.env_shape
        field = np.zeros((X_env, Y_env))
        self._map = np.zeros(self.env_shape)

        if self.foi is not None:
            self._map[:,:,0] = self.foi
            return

        np.random.seed(seed)

        for row in np.arange(1, X_env - 1):
            left = np.random.randint(1, X_env // 2)
            right = np.random.randint(X_env // 2, X_env - 1)
            field[row, left:right] = 1

        for col in np.arange(1, Y_env - 1):
            top = np.random.randint(1, Y_env // 2)
            bottom = np.random.randint(Y_env // 2, Y_env - 1)
            field[top:bottom, col] = 1

        self._map = np.zeros(self.env_shape)
        self._map[:, :, 0] = field

    def _drone_action(self, drone, action):
        X, Y, Z = self.env_shape
        x, y, z = self._drones[drone].pos
        if action == Action.LEFT:
            new_pos = (max(x - 1, 0), y, z)
        elif action == Action.RIGHT:
            new_pos = (min(x + 1, X - 1), y, z)
        elif action == Action.FORWARD:
            new_pos = (x, min(y + 1, Y - 1), z)
        elif action == Action.BACKWARD:
            new_pos = (x, max(y - 1, 0), z)
        elif action == Action.UP:
            new_pos = (x, y, min(z + 1, Z - 1))
        elif action == Action.DOWN:
            new_pos = (x, y, max(z - 1, 1))
        else:
            print('Invalid action.')

        all_pos = set([x.pos for x in self._drones.values()])
        if new_pos in all_pos:
            return
        self._drones[drone].pos = new_pos

    def _cost(self):
        pass

    def _compute_view_masks(self):
        ground = self._map[:, :, 0]
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in ground.shape])
        for drone in self._drones.values():
            view = np.full(ground.shape, False, dtype=bool)
            x_drone, y_drone = drone.pos[:2]
            z = drone.pos[2]
            for x, y in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z
                q = np.array([x, y])
                if all([
                    x > x_drone - x_proj,
                    x < x_drone + x_proj,
                    y > y_drone - y_proj,
                    y < y_drone + y_proj
                    ]):
                    view[x, y] = True
            drone.view = view

    def _gr(self):
        self._compute_view_masks()
        ground = self._map[:,:,0].astype(bool)

        covered = 0
        for drone in self._drones.values():
            covered += np.sum(drone.view.flatten() & ground.flatten())

        drone_set = set(self._drones.values())

        overlapped = 0
        if len(drone_set) > 1:
            for drone in self._drones.values():
                view = drone.view
                other_drones = drone_set - set((drone,))
                other_view = np.sum([x.view for x in other_drones], axis=0)
                overlapped += np.sum(view.flatten() & other_view.flatten())

        # if covered == sum(ground.flatten()) and overlapped == 0:
        #     return 0.1
        # else:
        #     return 0
        if covered == sum(ground.flatten()) and overlapped == 0:
            self._done = True
            return 0.1
        # return covered - overlapped
        return 0

    def _init_drones(self, seed=None):
        x, y, z = self.env_shape
        existing = set()
        np.random.seed(seed)
        for i in range(self.n_drones):
            pos = (np.random.randint(0, x), np.random.randint(0, y), np.random.randint(1, z))
            while pos in existing:
                pos = (np.random.randint(0, x), np.random.randint(0, y), np.random.randint(1, z))
            existing.add(pos)
            self._drones[i] = Drone(pos, self.fov)

    def _state(self):
        return [d.pos for d in self._drones.values()]