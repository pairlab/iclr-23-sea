import os
import joblib
import gym
import numpy as np
import time
import csv

from ..core.utils import preemptive_save


class CrafterMonitorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, monitor_id, save_path, save_freq=10):
        super(CrafterMonitorWrapper, self).__init__(env)
        self.monitor_id = monitor_id
        self.save_path = str(save_path) + f"/{monitor_id}_stats.csv"
        self.save_freq = max(1, save_freq)
        self.episode_step = 0
        self.last_save_timestamp = 0
        self.achv_names = []
        self.rows = []
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                reader = csv.reader(f)
                header = reader.__next__()
                assert header[0] == "timestamp"
                assert header[1] == "episode_steps"
                self.achv_names = header[2:]
                for row in reader:
                    self.rows.append(row)
        self.counter = dict()
        self.save()

    def save(self):
        rows = [["timestamp", "episode_steps"] + self.achv_names] + self.rows
        preemptive_save(rows, self.save_path, type="csv")

    def add_cols(self, achvs):
        self.achv_names += achvs
        for i in range(len(self.rows)):
            self.rows[i] += [0] * len(achvs)

    def add_row(self):
        new_cols = []
        for name in self.counter.keys():
            if name not in self.achv_names:
                new_cols.append(name)
        if len(new_cols) > 0:
            self.add_cols(new_cols)
        stats = [0] * len(self.achv_names)
        for name, count in self.counter.items():
            stats[self.achv_names.index(name)] = count
        row = [time.time(), self.episode_step] + stats
        self.rows.append(row)
        self.counter = dict()

    def reset(self):
        self.episode_step = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_step += 1
        # self.step_count += 1
        if done:
            achv = info["achievements"]
            for name, count in achv.items():
                if count >= 1:
                    if name not in self.counter:
                        self.counter[name] = 0
                    self.counter[name] += 1
            self.add_row()
            self.episode_step = 0
            if time.time() > self.last_save_timestamp + self.save_freq:
                self.save()
                self.last_save_timestamp = time.time()
        return observation, reward, done, info


class CrafterRenderWrapper(gym.Wrapper):
    def render(self, mode="human", **kwargs): 
        return self.env.render(**kwargs)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))
