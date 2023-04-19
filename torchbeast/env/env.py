import os
import sys
import joblib
import numpy as np
import torch
from copy import deepcopy

import gym


from .wrappers import ObjectiveWrapper, ObservationDictWrapper, WarpFrame, get_objective_wrapper
from .crafter_wrappers import CrafterRenderWrapper, ImageToPyTorch, CrafterMonitorWrapper


CRAFTER_KWARGS = dict(
    size=(84, 84),
    render_centering=False, 
    health_reward_coef=0.0, 
    immortal=True, 
    idle_death=100
)

CRAFTER_ORIGINAL_KWARGS = dict(
    size=(84, 84),
    render_centering=False,
    vanila=True
)

MINIGRID_KWARGS = dict(
    default=dict(achievement_reward=True),
    keycorridor=dict(room_size=5, num_rows=3),
    distractions_hard=dict(room_size=5, num_rows=9, num_nodes=15),
    distractions=dict(room_size=5, num_rows=5, num_nodes=8),
    distractions_easy=dict(room_size=5, num_rows=2, num_nodes=4, max_steps=300)
)


ENV_KWARGS = dict(
    crafter=CRAFTER_KWARGS,
    minigrid=MINIGRID_KWARGS,
)


def get_env(flags):
    return flags.env.split('-')[0]


def make_env(env_name, kwargs={}, flags=None, dummy_env=False):
    kwargs = deepcopy(kwargs)
    env_names = env_name.split('-')

    base_env = env_names[0]
    is_crafter = base_env == 'crafter'
    is_minigrid = base_env == 'minigrid'

    env = None

    if is_crafter:
        from crafter.env import Env as CrafterEnv
        env_cls = CrafterEnv
    elif is_minigrid:
        import gym_minigrid.envs
        from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        MINIGRID_ENVS = dict(
            keycorridor=gym_minigrid.envs.KeyCorridor,
            blockedunlockpickup=gym_minigrid.envs.BlockedUnlockPickup,
            distractions=gym_minigrid.envs.Distractions
        )
        env_cls = MINIGRID_ENVS[env_names[1]]
    else:
        raise NotImplementedError(f'Unrecognized env: {base_env}')

    def get_key(key, default=None):
        if key in kwargs:
            return kwargs.pop(key)
        elif flags is not None:
            return flags.get(key, default)
        else:
            return default

    env_id = get_key("env_id")
    crafter_monitor = get_key("use_crafter_monitor", False)

    if len(env_names) == 1:
        env_kwargs = deepcopy(ENV_KWARGS[base_env])
    else:
        env_kwargs = deepcopy(ENV_KWARGS[base_env].get('default', {}))
        if is_minigrid and env_names[1] == 'distractions':
            if get_key("distractions_hard", False):
                env_kwargs.update(ENV_KWARGS[base_env].get("distractions_hard", {}))
            elif get_key("distractions_easy", False):
                env_kwargs.update(ENV_KWARGS[base_env].get("distractions_easy", {}))
        else:
            env_kwargs.update(ENV_KWARGS[base_env].get(env_names[1], {}))
    
    if is_crafter:
        if get_key("crafter_original", False):
            env_kwargs = deepcopy(CRAFTER_ORIGINAL_KWARGS)
        if get_key("crafter_limited", False):
            env_kwargs["partial_achievements"] = "limited"
            env_kwargs["idle_death"] = 500

    num_objectives = get_key("num_objectives")
    objective_selection_algo = get_key("objective_selection_algo")
    causal_graph_load_path = get_key("causal_graph_load_path")
    include_new_tasks = get_key("include_new_tasks", True)
    done_if_reward = get_key("done_if_reward", False)
    graph_no_jumping = get_key("graph_no_jumping", False)
    env_kwargs.update(kwargs)

    if env is None:
        env = env_cls(**env_kwargs)

    if is_crafter:
        env = CrafterRenderWrapper(env)
        if crafter_monitor and env_id is not None:
            save_dir = get_key("savedir") + "/crafter_monitor"
            os.makedirs(save_dir, exist_ok=True)
            env = CrafterMonitorWrapper(env, env_id, save_dir, save_freq=30)
    if is_minigrid:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env = WarpFrame(env, grayscale=False)
    if is_crafter or is_minigrid:
        env = ImageToPyTorch(env)
        env = ObservationDictWrapper(env, 'frame')
    if num_objectives is not None:
        if objective_selection_algo == 'random':
            selection = ('random', {})
        elif causal_graph_load_path is not None:
            graph = joblib.load(causal_graph_load_path)
            selection = ('graph', {'graph': graph, 'no_jumping': graph_no_jumping})
        else:
            selection = ('random', {})
        env = get_objective_wrapper(selection[0], selection[1], env, num_objectives, include_new_tasks=include_new_tasks, done_if_reward=done_if_reward)

    return env
