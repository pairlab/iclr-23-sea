import gym
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml
import cv2
cv2.ocl.setUseOpenCL(False)
from ..core.utils import preemptive_save


class ActionSelectWrapper(gym.ActionWrapper):
    def action(self, action):
        return int(action)


class ObservationDictWrapper(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env, obs_key):
        super(ObservationDictWrapper, self).__init__(env)
        self.obs_key = obs_key
        old_shape = self.observation_space
        self.observation_space = gym.spaces.Dict({
            obs_key: old_shape
        })

    def observation(self, observation):
        return {
            self.obs_key: observation
        }


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


def ident(arr):
    return 1 if len(arr) > 0 else 0


class EventWrapper(gym.Wrapper):
    EVENTS = {
        'crafter': ['collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up']
    }
    def __init__(self, env: gym.Env, env_name='crafter'):
        super(EventWrapper, self).__init__(env)

        self.events = EventWrapper.EVENTS[env_name]
        self.n_events = len(self.events)
        self.observation_space = gym.spaces.Dict({
            'event': gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.n_events,),
                dtype=np.uint8
            ),
            **env.observation_space
        })

    def reset(self):
        return {
            'event': np.zeros(self.n_events, dtype=np.uint8),
            **self.env.reset()
        }
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        event_encoding = np.zeros(self.n_events, dtype=np.uint8)
        for event in info['events']:
            event_encoding[self.events.index(event)] = 1
        return {
            'event': event_encoding,
            **observation
        }, reward, done, info


class ObjectiveWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_objectives, include_new_tasks=True, done_if_reward=False, seed=None):
        super(ObjectiveWrapper, self).__init__(env)
        self.seed(seed)
        # self._random = np.random.RandomState(self._seed)

        self._n = num_objectives
        self._num_objectives = num_objectives + 1 if include_new_tasks else num_objectives
        self._include_new_tasks = include_new_tasks
        self._done_if_reward = done_if_reward
        self.observation_space = gym.spaces.Dict({
            "objective": gym.spaces.Box(
                low=0, 
                high=num_objectives,
                shape=(),
                dtype=np.uint8
            ),
            "completed": gym.spaces.Box(
                low=0, 
                high=1,
                shape=(self._num_objectives,),
                dtype=np.uint8
            ),
            "last_completed": gym.spaces.Box(
                low=0, 
                high=num_objectives,
                shape=(),
                dtype=np.uint8
            ),
            "regen_steps": gym.spaces.Box(
                low=0,
                high=1000000,
                shape=(),
                dtype=np.int32
            ),
            **env.observation_space.spaces
        })
        self._remaining = None
        self._objective = None
        self._last_completed = None
        self._completed = np.zeros((self._num_objectives,), dtype=np.uint8)
        self._done = False
        self._regen_steps = 0

    def seed(self, _seed=None):
        seed = np.random.randint(0, 2 ** 31 - 1) if _seed is None else _seed
        self._seed = seed
        self._random = np.random.default_rng(seed)
        self.env.seed(abs(hash(f"objective-sub-env-{seed}")) % (2 ** 32))

    def _init(self):
        # self._completed.zero_()
        self._seq_pos = 0
        self._completed[:] = 0
        self._remaining = set(range(self._num_objectives))
        self._last_completed = self._num_objectives

    def _generate_objective(self):
        pass

    def _regenerate_objective(self):
        self._regen_steps = 0
        self._objective = self._generate_objective()
        if self._objective is None:
            self._objective = self._random.integers(self._num_objectives)

    def _make_observation(self, observation):
        # print(self._objective, self._completed)
        return {
            "objective": np.array([self._objective], dtype=np.uint8),
            "completed": self._completed,
            "last_completed": np.array([self._last_completed], dtype=np.uint8),
            "regen_steps": np.array([self._regen_steps], dtype=np.int32),
            **observation
        }

    def reset(self):
        self._done = False
        self._init()
        self._regenerate_objective()
        return self._make_observation(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._regen_steps += 1
        info['reset_done'] = done
        if self._done_if_reward:
            done = done or (reward > 0.1)
        return self._make_observation(observation), reward, done, info

    def complete_objective(self, objective, regenerate=True):
        if objective < self._num_objectives:
            self._completed[objective] = 1
        self._last_completed = objective
        if objective in self._remaining:
            self._remaining.remove(objective)
        if regenerate == "correct":
            if objective == self._objective:
                self._regenerate_objective()
        elif regenerate:
            self._regenerate_objective()


class GraphObjectiveWrapper(ObjectiveWrapper):
    def __init__(self, env: gym.Env, num_objectives, graph, no_jumping=False, **kwargs):
        super().__init__(env, num_objectives, **kwargs)
        self._graph = graph
        self._no_jumping = no_jumping

    def _generate_objective(self):
        cont_choices = []
        jump_choices = []
        nghb_choices = []
        unrd_choices = []
        for u in range(self._n):
            if self._completed[u]:
                continue
            ready = True
            for v in range(self._n):
                if self._graph[v][u] and not self._completed[v]:
                    ready = False
                    break
            if ready:
                if self._last_completed < self._n:
                    x = self._last_completed
                    if self._graph[x][u]:
                        cont_choices.append(u)
                    else:
                        found = False
                        for y in range(self._n):
                            if not self._completed[y] and self._graph[x][y] and self._graph[u][y]:
                                found = True
                                break
                        if found:
                            nghb_choices.append(u)
                        else:
                            jump_choices.append(u)
                else:
                    jump_choices.append(u)
            else:
                unrd_choices.append(u)
        if self._include_new_tasks and self._random.random() < 0.4:
            objective = self._num_objectives - 1
        else:
            lc = ident(cont_choices)
            ln = ident(nghb_choices)
            lj = ident(jump_choices)
            lu = ident(unrd_choices)
            if self._no_jumping:
                if lc == 0:
                    if lj == 0:
                        objective = None
                    else:
                        objective = self._random.choice(jump_choices)
                else:
                    objective = self._random.choice(cont_choices)
            else:
                wc = 80 * lc
                wn = 10 * ln
                wj = 10 * lj
                wu = 5 * lu
                if wc + wn + wj + wu == 0:
                    objective = None
                else:
                    p = np.array([wc, wn, wj, wu])
                    choices = [cont_choices, nghb_choices, jump_choices, unrd_choices]
                    while True:
                        c = self._random.choice(4, p=p / p.sum())
                        if len(choices[c]) > 0:
                            objective = self._random.choice(choices[c])
                            break
        return objective


class FixedObjectiveWrapper(ObjectiveWrapper):
    def __init__(self, env: gym.Env, num_objectives, objective, **kwargs):
        super().__init__(env, num_objectives, **kwargs)
        self._fixed_objective = objective

    def _generate_objective(self):
        return self._fixed_objective
    

class RandomObjectiveWrapper(ObjectiveWrapper):
    def __init__(self, env: gym.Env, num_objectives, **kwargs):
        super().__init__(env, num_objectives, **kwargs)
    def _generate_objective(self):
        if len(self._remaining) > 0:
            return self._random.choice(list(self._remaining))
        else:
            return None
        

class SequenceObjectiveWrapper(ObjectiveWrapper):
    def __init__(self, env: gym.Env, num_objectives, sequence, **kwargs):
        super().__init__(env, num_objectives, **kwargs)
        self._sequence = sequence

    def _generate_objective(self):
        for ob in self._sequence:
            if not self._completed[ob]:
                return ob
        return None
    

def get_objective_wrapper(algo, params, env: gym.Env, num_objectives, **kwargs):
    return {
        "random": RandomObjectiveWrapper,
        "graph": GraphObjectiveWrapper,
        "fixed": FixedObjectiveWrapper,
        "seq": SequenceObjectiveWrapper
    }[algo](env, num_objectives, **params, **kwargs)
