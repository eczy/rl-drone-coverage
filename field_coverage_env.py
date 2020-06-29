import gym
import numpy as np
from enum import Enum

class FieldCoverageEnv(gym.Env):
    """Field coverage environment for multi-agent cooperative reinforcement learning."""

    class Action(Enum):
        LEFT = 0
        RIGHT = 1
        FORWARD = 2
        BACKWARD = 3
        UP = 4
        DOWN = 5

    class Drone(object):
        def __init__(self, pos, fov):
            self.fov = fov
            self.pos = pos

    def __init__(self, shape, foi, fov, n_drones, max_steps=2000, seed=None):
        super().__init__()
        assert len(shape) == 3, 'Environmnet shape must be of form (X, Y, Z).'
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Box(low=0, high=np.array(shape))
        self.shape = shape
        self.fov = fov
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.foi = foi

        self._drones = {}
        self._steps = 0

        self._map = np.zeros(self.shape)
        self._map[:,:,0] = self.foi

    def reset(self):
        self._steps = 0
        X, Y, Z = self.shape
        drones = {}
        positions = set()
        for i in range(self.n_drones):
            while True:
                pos = np.random.randint(0, X), np.random.randint(0, Y), np.random.randint(1, Z)
                if pos not in drones.values():
                    break
            positions.add(pos)
            drones[i] = self.Drone(pos, self.fov)
        self._drones = drones
        return self._state()

    def step(self, action):
        assert len(action) == len(self._drones), 'Joint action must be defined for each agent.'
        for drone, a in action.items():
            self._move_drone(drone, self.Action(a))
        observation = self._state()
        reward = self._reward()
        success = reward > 0
        done = success or self._steps == self.max_steps
        return observation, reward, done, {'success': success}

    def _state(self):
        return [x.pos for x in self._drones.values()]

    def _reward(self):
        masks = self._view_masks()
        foi = self.foi.astype(int)
        coverage = 0
        for i, drone in self._drones.items():
            coverage += np.sum(masks[i].flatten() & foi.flatten())
        
        drones = set(self._drones.keys(),)
        overlap = 0
        if len(drones) > 1:
            for i, drone in self._drones.items():
                mask = masks[i]
                other_masks = np.sum([masks[x] for x in drones - {i}], axis=0)
                overlap += np.sum(mask.flatten() & other_masks.flatten())
        if coverage == sum(foi.flatten()) and overlap == 0:
            return 0.1
        return 0


    def _view_masks(self):
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.foi.shape])
        view_masks = {}
        for i, drone in self._drones.items():
            mask = np.zeros_like(self.foi).astype(int)
            x, y, z = drone.pos
            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z
                if all([
                    xc > x - x_proj,
                    xc < x + x_proj,
                    yc > y - y_proj,
                    yc < y + y_proj
                ]):
                    mask[xc, yc] = True
            view_masks[i] = mask
        return view_masks
    
    def _move_drone(self, drone, action):
        X, Y, Z = self.shape
        x, y, z = self._drones[drone].pos

        if action == self.Action.LEFT:
            new_pos = max(x - 1, 0), y, z
        elif action == self.Action.RIGHT:
            new_pos = min(x + 1, X - 1), y, z
        elif action == self.Action.FORWARD:
            new_pos = x, min(y + 1, Y - 1), z
        elif action == self.Action.BACKWARD:
            new_pos = x, max(y - 1, 0), z
        elif action == self.Action.UP:
            new_pos = x, y, min(z + 1, Z - 1)
        elif action == self.Action.DOWN:
            new_pos = x, y, max(z - 1, 1)
        else:
            raise ValueError(f'Invalid action {action} for agent {drone}')

        if new_pos in set(self._state()):
            return
        self._drones[drone].pos = new_pos