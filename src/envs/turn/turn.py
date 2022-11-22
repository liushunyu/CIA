import numpy as np
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th
import random
import logging.config

N_ACTIONS = 6  # 0 up, 1 down, 2 left, 3 right, 4 eat, 5 stay
N_AGENTS = 2
HEIGHT = 5
WIDTH = 5
delta_xy = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0]]


class coordinate(object):
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class TurnEnv(MultiAgentEnv):

    def __init__(
            self,
            map_name='turn',
            time_limit=100,
            is_token=False,
            seed=None
    ):
        self.is_token = is_token
        self._seed = seed
        np.random.seed(self._seed)
        self.n_enemies = 0
        self.episode_limit = time_limit
        self.time_limit = time_limit
        self.env_name = map_name
        self.time_step = 0
        self.n_actions = N_ACTIONS
        self.n_agents = N_AGENTS
        self.obs = np.array([])
        self.zero_matrix = np.zeros((5, 5))
        self.agent_pos = [coordinate(0, 0), coordinate(4, 4)]
        self.key_X = np.random.randint(low=0, high=HEIGHT)
        self.key_Y = np.random.randint(low=0, high=WIDTH)
        self.is_alive = [1, 0]

    def calc(self, action):
        res = 0.0
        change = False
        
        for id in range(2):
            if self.is_alive[id] == 0:
                if (action[id] != 5):
                    res += -1
            else:
                if action[id] == 4 and self.agent_pos[id].X == self.key_X and self.agent_pos[id].Y == self.key_Y:
                    res += 10.0
                    change = True

                dest_X = self.agent_pos[id].X + delta_xy[action[id]][0]
                dest_Y = self.agent_pos[id].Y + delta_xy[action[id]][1]

                if dest_X < 0 or dest_X > 4 or dest_Y < 0 or dest_Y > 4:
                    res += -1
                else:
                    self.agent_pos[id].X = dest_X
                    self.agent_pos[id].Y = dest_Y

        if change == True:
            self.is_alive[0] ^= 1
            self.is_alive[1] ^= 1
            self.key_X = np.random.randint(low=0, high=HEIGHT)
            self.key_Y = np.random.randint(low=0, high=WIDTH)

        return res

    def step(self, action):
        reward = self.calc(action)
        self.time_step += 1
        done = False

        if self.time_step >= self.time_limit:
            # done only if time steps exceed limitation
            done = True

        info = {}

        return reward, done, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        key_map = np.zeros((3, 3))
        agent_map = np.zeros((3, 3))
        if max(abs(self.agent_pos[agent_id ^ 1].X - self.agent_pos[agent_id].X), abs(self.agent_pos[agent_id ^ 1].Y - self.agent_pos[agent_id].Y)) <= 1:
            agent_map[(self.agent_pos[agent_id ^ 1].X - self.agent_pos[agent_id].X), (self.agent_pos[agent_id ^ 1].Y - self.agent_pos[agent_id].Y)] = 1.0
        if max(abs(self.key_X - self.agent_pos[agent_id].X), abs(self.key_Y - self.agent_pos[agent_id].Y)) <= 1:
            key_map[(self.key_X - self.agent_pos[agent_id].X), (self.key_Y - self.agent_pos[agent_id].Y)] = 1.0
        return agent_map.reshape(-1).tolist() + key_map.reshape(-1).tolist()

    def get_obs_size(self):
        """Returns the size of the observation."""
        return (3 * 3) * 2

    def get_state(self):
        """Returns the global state."""
        key_map = np.zeros((5, 5))
        key_map[self.key_X, self.key_Y] = 1.0
        agent0_map = np.zeros((5, 5))
        agent0_map[self.agent_pos[0].X, self.agent_pos[0].Y] = 1.0
        agent1_map = np.zeros((5, 5))
        agent1_map[self.agent_pos[1].X, self.agent_pos[1].Y] = 1.0

        if self.is_token == False:
            return key_map.reshape(-1).tolist() + agent0_map.reshape(-1).tolist() + agent1_map.reshape(-1).tolist() + self.is_alive
        else:
            return (key_map.reshape(-1).tolist() + agent0_map.reshape(-1).tolist() + agent1_map.reshape(-1).tolist() + self.is_alive) * self.n_agents

    def get_state_size(self):
        """Returns the size of the global state."""
        tmp = 5 * 5 + 5 * 5 + 5 * 5 + 2
        if (self.is_token == False):
            return tmp
        else:
            return tmp * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self.agent_pos = [coordinate(0, 0), coordinate(4, 4)]
        self.key_X = np.random.randint(low=0, high=HEIGHT)
        self.key_Y = np.random.randint(low=0, high=WIDTH)
        self.is_alive = [1, 0]
        self.time_step = 0
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["unit_dim"] = 5 * 5 * 3 + 2

        return env_info
