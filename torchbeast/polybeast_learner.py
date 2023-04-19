# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run with OMP_NUM_THREADS=1.
#

from base64 import encode
import collections
import logging
import os
import threading
import time
import timeit
import traceback

import wandb
import omegaconf
import nest
import torch
import numpy as np
import joblib
import pathlib

import libtorchbeast

from .core import file_writer
from .core import vtrace
from .core.utils import preemptive_save
from .core.clustering import cluster

from .models import create_model, create_pred_model, create_rew_classifier, create_rnd_net
from .models.baseline import NetHackNet, BaselineNet, PredictionModel, RewardClassifier
from .env import make_env

from torch import nn
from torch.nn import functional as F


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def color_message(message, color):
    color_code = dict(
        grey = "\x1b[38;20m",
        yellow = "\x1b[33;20m",
        red = "\x1b[31;20m",
        bold_red = "\x1b[31;1m"
    )
    reset = "\x1b[0m"
    return color_code[color] + message + reset


def compute_baseline_loss(diff):
    if len(diff.shape) > 2:
        return 0.5 * torch.sum(diff ** 2, dim=tuple(range(2, len(diff.shape))))
    else:
        return 0.5 * diff ** 2


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach()
    return torch.sum(policy_gradient_loss_per_timestep)


def batch_select(tensor, index):
    # tensor: [T, B, K, ...]
    # index: [T, B]
    batch_shape = index.shape
    n_batch_dim = len(batch_shape)
    assert tensor.shape[:n_batch_dim] == index.shape[:n_batch_dim]
    tensor_shape = tensor.shape[n_batch_dim + 1:]
    t_index = index.view(*batch_shape, 1, *((1,) * len(tensor_shape))).expand(*batch_shape, 1, *tensor_shape)
    return torch.gather(tensor, n_batch_dim, t_index).squeeze(n_batch_dim)


# TODO(heiner): Given that our nest implementation doesn't support
# namedtuples, using them here doesn't seem like a good fit. We
# probably want to nestify the environment server and deal with
# dictionaries?
EnvOutput = collections.namedtuple(
    "EnvOutput", "observation last_observation last_action rewards done reset_done episode_step episode_return episode_server episode_count"
)
PRIME = int(1e9 + 7)
AgentOutput = NetHackNet.AgentOutput
# ExtendedAgentOutput = NetHackNet.ExtendedAgentOutput
Batch = collections.namedtuple("Batch", "env agent")


def calc_objective(env_output: EnvOutput, flags):
    if "objective" in env_output.observation:
        return env_output.observation["objective"][..., 0].to(dtype=torch.int64)
    else:
        return None


def encode_combination(completed: np.ndarray):
    K = completed.shape[0]
    encoded = 0
    for i in range(K):
        encoded ^= completed[i] << i
    return encoded


def _clean(ts):
    return ts.clone().detach().cpu().numpy()


def clean_multi(*args):
    return (_clean(v) for v in args)


def inference(
    inference_batcher, model, rew_classifier, replay_buffer, flags, actor_device, lock=threading.Lock()
):  # noqa: B008

    multi_objective = flags.multi_objective

    if multi_objective:
        num_objectives = flags.num_objectives
        if flags.include_new_tasks:
            num_objectives += 1
            
        mo_mean_steps = np.zeros((num_objectives + 1, num_objectives), dtype=np.float32)
        mo_counter = np.zeros((num_objectives + 1, num_objectives), dtype=np.int32)
        mo_succ_counter = np.zeros((num_objectives + 1, num_objectives), dtype=np.int32)

    last_save = time.time()
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            env_outputs = EnvOutput._make(batched_env_outputs)
            observation = env_outputs.observation
            done = env_outputs.done
            reset_done = env_outputs.reset_done
            observation["done"] = done
            observation["reset_done"] = reset_done
            last_action = env_outputs.last_action
            last_observation = env_outputs.last_observation
            observation, agent_state, reset_done, last_action, last_observation = nest.map(
                lambda t: t.to(actor_device, non_blocking=True),
                (observation, agent_state, reset_done, last_action, last_observation),
            )

            with lock:
                outputs = model(observation, agent_state)
                if multi_objective:
                    classes, embeds = rew_classifier(observation, last_action.to(torch.long), reset_done, last_obs=last_observation, return_embeds=True)

            core_outputs, agent_state = nest.map(lambda t: t.cpu(), outputs)
            T, B, *_ = core_outputs["action"].shape
            # Restructuring the output in the way that is expected
            # by the functions in actorpool.
            agent_outputs = tuple(
                (
                    core_outputs["action"],
                    core_outputs["policy_logits"],
                    core_outputs["baseline"]
                )
            )

            if multi_objective:
                objective = calc_objective(env_outputs, flags)

                for i in range(T):
                    for j in range(B):
                        obj = env_outputs.last_observation["objective"][i][j].item()
                        last_comp = env_outputs.last_observation["last_completed"][i][j].item()
                        finished = classes[i][j].item()
                        regen_steps = env_outputs.last_observation["regen_steps"][i][j].item()
                        if env_outputs.rewards[i][j].item() > 0.1:
                            if finished == obj:
                                m = mo_succ_counter[last_comp][obj]
                                mo_mean_steps[last_comp][obj] = (m * mo_mean_steps[last_comp][obj] + regen_steps) / (m + 1)
                                mo_succ_counter[last_comp][obj] += 1

                if time.time() - last_save > 60:
                    preemptive_save((mo_counter, mo_succ_counter, mo_mean_steps), os.path.join(flags.savedir, "mo_stats.obj"), type="joblib")
                    last_save = time.time()

                if not flags.objective_as_input:
                    agent_outputs = nest.map(lambda t: batch_select(t, objective), agent_outputs)
                classes = classes.to(device="cpu")
                complete = torch.gt(env_outputs.rewards, 0.1).to(torch.int64)
                correct = torch.eq(classes, objective).to(torch.int64)
                non_leaf = torch.ones_like(classes, dtype=torch.bool)
                non_leaf = non_leaf.to(torch.int64)
                save = torch.zeros_like(complete)
                done_com = correct * complete
                reset = torch.zeros_like(complete)
                command = classes * 16 + done_com * 8 + complete * 4 + save * 2 + reset
            else:
                command = torch.zeros((T, B), dtype=torch.int64)

            outputs = (
                agent_outputs,
                agent_state,
                command
            )
            batch.set_outputs(outputs)

            if replay_buffer is not None:
                for b in range(B):
                    for t in range(T):
                        replay_buffer.add(*clean_multi(observation["frame"][t][b], core_outputs["action"][t][b], env_outputs.rewards[t][b], done[t][b]))


def get_env_tuple(env_outputs: EnvOutput, i, j):
    episode_server = env_outputs.episode_server[i, j].item()
    episode_count = env_outputs.episode_count[i, j].item()
    episode_step = env_outputs.episode_step[i, j].item()
    return episode_server, episode_count, episode_step
    

class StepCounter:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.set_storage = set()
        self.deque_storage = collections.deque()
        self.cnt = 0

    def check_exists(self, item):
        return item in self.set_storage

    def check_size(self):
        while len(self.deque_storage) > self.maxlen:
            to_pop = self.deque_storage.popleft()
            self.set_storage.remove(to_pop)
    
    def add(self, item):
        self.set_storage.add(item)
        self.deque_storage.append(item)
        self.check_size()
        self.cnt += 1

    def check_add(self, item):
        if not self.check_exists(item):
            self.add(item)
            return False
        else:
            return True


def check_register(ident, registered: StepCounter):
    return registered.check_add(ident)


STEPS_MAXLEN = int(1e7)


def learn(
    learner_queue: libtorchbeast.BatchingQueue,
    model: BaselineNet,
    actor_model: BaselineNet,
    pred_model: PredictionModel,
    optimizer: torch.optim.Optimizer,
    pred_optimizer: torch.optim.Optimizer,
    rnd_nets,
    rnd_optimizer,
    rew_classifier: RewardClassifier,
    clustering_buffers,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    stats,
    flags,
    plogger,
    learner_device,
    lock=threading.Lock(),  # noqa: B008
):
    multi_objective = flags.multi_objective

    episode_index = dict()
    key_frames_maxlen = flags.key_frames_maxlen
    active_indices = collections.deque(maxlen=key_frames_maxlen)
    reward_registered_steps = StepCounter(STEPS_MAXLEN)
    done_registered_steps = StepCounter(STEPS_MAXLEN)
    return_registered_steps = StepCounter(STEPS_MAXLEN)
    returns = collections.defaultdict(float)
    key_frames = dict(obs=dict(), last_obs=dict(), action=[], done=[])
    obs_keys = []
    episode_cnt = 0

    pred = pred_model is not None
    do_clustering = flags.do_clustering and pred

    if do_clustering:
        clustering_embeddings, clustering_indices = clustering_buffers
        # clustering_embeddings = []
        # clustering_indices = []
        clustering_maxlen = flags.clustering_maxlen
        # stats["clustered"] = False

    rnd = rnd_optimizer is not None
    rnd_truth_net, rnd_pred_net = rnd_nets

    step_cnt = 0

    for tensors in learner_queue:
        if multi_objective:
            objective_episode_returns = [[] for _ in range(flags.num_objectives + 1)]

        tensors = nest.map(lambda t: t.to(learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        observation, last_observation, last_action, reward, done, reset_done, *_ = env_outputs
        env_outputs = EnvOutput._make(env_outputs)

        if len(obs_keys) == 0:
            obs_keys = list(observation.keys())
            for key in obs_keys:
                key_frames['obs'][key] = []
                key_frames['last_obs'][key] = []

        observation["reward"] = reward
        observation["done"] = done
        observation["reset_done"] = reset_done

        lock.acquire()  # Only one thread learning at a time.

        # PREDICTION MODEL
        if pred:

            env_outputs = EnvOutput._make(env_outputs)
            actor_outputs = AgentOutput._make(actor_outputs)
            env_rewards = env_outputs.rewards.clone()
            T, B, *_ = env_rewards.shape

            pred_outputs, pred_model_loss, _, pred_model_embeds = pred_model(observation, actor_outputs.action, done, last_obs=last_observation)

            if not flags.no_reward_pred:
                prediction_loss = 0
                for key, prediction in pred_outputs.items():
                    if key == "reward":
                        truth = torch.gt(torch.abs(env_rewards), 0.02).float()[..., None]
                    _prediction_loss = compute_baseline_loss(
                        prediction - truth
                    )
                    pl = _prediction_loss.sum().item()
                    stats[f"{key}_predition_loss"] = pl
                    prediction_loss += _prediction_loss.sum()
                for key, loss in pred_model_loss.items():
                    prediction_loss += loss
                    stats[f"pred_model_{key}"] = loss.item()
            else:
                prediction_loss = 0

            if type(prediction_loss) != int:
                stats["prediction_loss"] = prediction_loss.item()
            
            contrast_reward = torch.gt(torch.abs(env_rewards), 0.2).float()

            if not flags.no_contrast_loss:
                if flags.contrast_step_limit is None or step_cnt <= flags.contrast_step_limit:
                    for i, j in np.argwhere(contrast_reward.detach().cpu().numpy() > 0.5):
                        episode_server, episode_count, episode_step = get_env_tuple(env_outputs, i, j)
                        if check_register((episode_server, episode_count, episode_step), reward_registered_steps):
                            continue
                        episode_hash = (episode_server, episode_count)
                        if episode_hash not in episode_index:
                            episode_index[episode_hash] = episode_cnt
                            episode_cnt += 1
                            for key in obs_keys:
                                key_frames['obs'][key].append([])
                                key_frames['last_obs'][key].append([])
                            key_frames['action'].append([])
                            key_frames['done'].append([])
                        index = episode_index[episode_hash]
                        for key in obs_keys:
                            if type(key_frames['obs'][key][index]) != list:
                                logging.warning(f"not a list {key} {index} {(episode_server, episode_count, episode_step)}")
                            key_frames['obs'][key][index].append(observation[key][i, j].clone())
                            if i > 0:
                                key_frames['last_obs'][key][index].append(observation[key][i - 1, j].clone())
                            else:
                                key_frames['last_obs'][key][index].append(torch.zeros_like(observation[key][i, j]))
                        key_frames['action'][index].append(actor_outputs.action[i, j].clone())
                        key_frames['done'][index].append(reset_done[i, j].clone())

                        if do_clustering and len(clustering_embeddings) < clustering_maxlen:
                            clustering_embeddings.append(_clean(pred_model_embeds[i, j].clone()))
                            clustering_indices.append((index, episode_step))
                            if len(clustering_embeddings) % 100 == 0:
                                logging.info(f"len(clustering_embeddings): {len(clustering_embeddings)}, {step_cnt}")
                    
                    for i, j in np.argwhere(reset_done.detach().cpu().numpy() > 0.5):
                        episode_server, episode_count, episode_step = get_env_tuple(env_outputs, i, j)
                        if check_register((episode_server, episode_count, episode_step), done_registered_steps):
                            continue
                        if episode_step == 0:
                            continue
                        episode_hash = (episode_server, episode_count)
                        if reset_done[i, j].item() and episode_hash in episode_index:
                            index = episode_index[episode_hash]
                            if len(key_frames['action'][index]) >= 2:
                                if len(active_indices) == key_frames_maxlen:
                                    to_pop = active_indices[0]
                                    for key in obs_keys:
                                        key_frames['obs'][key][to_pop] = None
                                        key_frames['last_obs'][key][to_pop] = None
                                    key_frames['action'][to_pop] = None
                                    key_frames['done'][to_pop] = None
                                else:
                                    to_pop = None
                                active_indices.append(index)
                                assert type(key_frames['action'][index]) == list
                                for key in obs_keys:
                                    key_frames['obs'][key][index] = torch.stack(key_frames['obs'][key][index], dim=0)
                                    key_frames['last_obs'][key][index] = torch.stack(key_frames['last_obs'][key][index], dim=0)
                                key_frames['action'][index] = torch.stack(key_frames['action'][index], dim=0)
                                key_frames['done'][index] = torch.stack(key_frames['done'][index], dim=0)

                # logging.info(f"len(active_indices): {len(active_indices)}, {step_cnt}")

                contrast_loss = 0
                contrast_batch_size = 128
                average_reward_length = 0
                # if True:
                if len(active_indices) > 5:
                    key_frames_batch = dict(obs={ key: [] for key in obs_keys }, last_obs={ key: [] for key in obs_keys }, action=[], done=[])
                    indices = np.random.choice(active_indices, contrast_batch_size, replace=True).tolist()
                    episode_indices = []
                    lengths = []
                    for index in indices:
                        L = key_frames['action'][index].shape[0]
                        c = min(L, 8)
                        # c = L
                        idx = np.random.choice(L, c, replace=False)
                        # logging.warning(str(idx))
                        episode_indices.append(idx.copy())
                        for key in obs_keys:
                            key_frames_batch['obs'][key].append(key_frames['obs'][key][index][idx])
                            key_frames_batch['last_obs'][key].append(key_frames['last_obs'][key][index][idx])
                        key_frames_batch['action'].append(key_frames['action'][index][idx])
                        key_frames_batch['done'].append(key_frames['done'][index][idx])
                        lengths.append(c)

                    for key in obs_keys:
                        key_frames_batch['obs'][key] = torch.cat(key_frames_batch['obs'][key], 0).unsqueeze(0)
                        key_frames_batch['last_obs'][key] = torch.cat(key_frames_batch['last_obs'][key], 0).unsqueeze(0)
                    key_frames_batch['action'] = torch.cat(key_frames_batch['action'], 0).unsqueeze(0)
                    key_frames_batch['done'] = torch.cat(key_frames_batch['done'], 0).unsqueeze(0)

                    _, _, _, embeds = pred_model(key_frames_batch['obs'], key_frames_batch['action'], key_frames_batch['done'], key_frames_batch['last_obs'])
                    embeds = embeds[0]
                    pos = 0
                    for index, L in zip(indices, lengths):
                        _embeds = embeds[pos: pos + L]
                        pos += L
                        in_dis = (_embeds.unsqueeze(0) - _embeds.unsqueeze(1)).square().sum(-1)
                        d = torch.exp(-in_dis / in_dis.max() * 2)
                        c = torch.det(d)
                        contrast_loss -= c
                        average_reward_length += L / contrast_batch_size
                    
                    stats["contrast_loss"] = contrast_loss.item() / contrast_batch_size
                    stats["average_reward_length"] = average_reward_length

                    del key_frames_batch
            else:
                contrast_loss = 0

            pred_total_loss = contrast_loss * 20 + prediction_loss

            if type(pred_total_loss) != int and not flags.no_train_pred:
                pred_optimizer.zero_grad()
                pred_total_loss.backward()
                if flags.grad_norm_clipping > 0:
                    nn.utils.clip_grad_norm_(pred_model.parameters(), flags.grad_norm_clipping)
                pred_optimizer.step()

                stats["pred_total_loss"] = pred_total_loss.item()

            # if do_clustering and not stats["clustered"] and len(clustering_embeddings) == clustering_embeddings_maxlen:
            #     logging.info("Collected enough data. Start clustering!")
            #     cluster(clustering_embeddings, clustering_indices)
            #     stats["clustered"] = True

        output, _ = model(observation, initial_agent_state, learning=True)

        # Use last baseline value (from the value function) to bootstrap.
        learner_outputs = AgentOutput._make(
            (output["action"], output["policy_logits"], output["baseline"])
        )

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in `batch` and `learner_outputs` at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        # Note that the env_outputs.frame is now a dict with 'features' and 'glyphs'
        # instead of actually being the frame itself. This is currently not a problem
        # because we never use actor_outputs.frame in the rest of this function.
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        if multi_objective:
            last_objective = env_outputs.last_observation["objective"][..., 0].to(dtype=torch.int64)
            if not flags.objective_as_input:
                learner_outputs = nest.map(lambda t: batch_select(t, last_objective), learner_outputs)

        learner_outputs = AgentOutput._make(learner_outputs)
        
        rewards = env_outputs.rewards[..., None]

        if multi_objective:
            with torch.no_grad():
                class_one_hot = rew_classifier(env_outputs.observation, env_outputs.last_action.to(dtype=torch.int64), env_outputs.reset_done, last_obs=env_outputs.last_observation, one_hot=True)
                rewards = rewards * class_one_hot

        if multi_objective:
            T, B, *_ = rewards.shape
            for i in range(T):
                for j in range(B):
                    ep_server, ep_count, ep_step = get_env_tuple(env_outputs, i, j)
                    if check_register((ep_server, ep_count, ep_step), return_registered_steps):
                        continue
                    ep_hash = (ep_server, ep_count)
                    if multi_objective:
                        returns[ep_hash] += rewards[i, j, last_objective[i, j].item()].item()
                    else:
                        returns[ep_hash] += rewards[i, j].item()
                    if env_outputs.done[i, j].item():
                        if multi_objective:
                            objective_episode_returns[last_objective[i, j].item()].append(returns[ep_hash])
                        del returns[ep_hash]

        if rnd:
            rnd_truth_value = rnd_truth_net(observation).detach()
            rnd_pred_value = rnd_pred_net(observation)
            rnd_error = compute_baseline_loss(rnd_truth_value - rnd_pred_value)
            rewards += rnd_error[1:, :, None]
            rnd_loss = rnd_error.sum()
        else:
            rnd_loss = 0

        if flags.clip_reward:
            rewards = torch.clip(rewards, min=-1, max=1)

        if flags.normalize_reward:
            model.update_running_moments(rewards)
            rewards /= model.get_running_std()

        if multi_objective:
            rewards = batch_select(rewards, last_objective) + flags.mo_expl_coef * rewards.sum(-1)
        else:
            rewards = rewards[..., 0]

        total_loss = 0

        # STANDARD EXTRINSIC LOSSES / REWARDS
        if flags.entropy_cost > 0:
            entropy_loss = flags.entropy_cost * compute_entropy_loss(
                learner_outputs.policy_logits
            )
            total_loss += entropy_loss

        discounts = (~env_outputs.done).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs.baseline,
            bootstrap_value=learner_outputs.baseline[-1],
        )

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        ).sum()
        total_loss += pg_loss + baseline_loss

        # BACKWARD STEP
        if not flags.no_train_actor:
            optimizer.zero_grad()
            total_loss.backward()
            if flags.grad_norm_clipping > 0:
                nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
            optimizer.step()
            scheduler.step()

            actor_model.load_state_dict(model.state_dict())

            if rnd:
                rnd_optimizer.zero_grad()
                rnd_loss.backward()
                rnd_optimizer.step()
                stats["rnd_loss"] = rnd_loss.item()

        # LOGGING
        if multi_objective:
            for i, ret in enumerate(objective_episode_returns):
                if len(ret) > 0:
                    stats[f"objective_return/{rew_classifier.cluster_names[i]}"] = np.mean(ret)

        step_cnt += flags.unroll_length * flags.batch_size

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode"] = stats.get("episode", 0) + len(episode_returns)
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        if flags.entropy_cost > 0:
            stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        # Only logging if at least one episode was finished
        if len(episode_returns):
            # TODO: log also SPS
            plogger.log(stats)
            if flags.wandb:
                wandb.log(stats, step=stats["step"])

        lock.release()


def train(flags):
    logging.info("Logging results to %s", flags.savedir)
    if isinstance(flags, omegaconf.DictConfig):
        flag_dict = omegaconf.OmegaConf.to_container(flags)
    else:
        flag_dict = vars(flags)
    plogger = file_writer.FileWriter(xp_args=flag_dict, rootdir=flags.savedir)

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        learner_device = torch.device(flags.learner_device)
        actor_device = torch.device(flags.actor_device)
    else:
        logging.info("Not using CUDA.")
        learner_device = torch.device("cpu")
        actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static. We could make it dynamic, but that
    # requires a loss (and learning rate schedule) that's batch size
    # independent.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    logging.info("Using model %s", flags.model)

    model = create_model(flags, learner_device)

    plogger.metadata["model_numel"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logging.info("Number of model parameters: %i", plogger.metadata["model_numel"])

    actor_model = create_model(flags, actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=model.initial_state(),
        num_labels=flags.num_objectives + 1,
    )

    replay_buffer = None

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if flags.pred_model:
        pred_model = create_pred_model(flags, learner_device)
        plogger.metadata["pred_model_numel"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        logging.info("Number of prediction model parameters: %i", plogger.metadata["pred_model_numel"])

        pred_optimizer = torch.optim.Adam(pred_model.parameters(), lr=1e-4, eps=1e-5)
    else:
        pred_model = None
        pred_optimizer = None

    if flags.use_rnd:
        rnd_truth_net, rnd_pred_net = create_rnd_net(flags, learner_device)
        rnd_optimizer = torch.optim.Adam(rnd_pred_net.parameters(), lr=1e-4, eps=1e-5)
    else:
        rnd_truth_net = None
        rnd_pred_net = None
        rnd_optimizer = None

    if flags.actor_load_dir and os.path.exists(flags.actor_load_dir):
        logging.info("Loading actor: %s" % flags.actor_load_dir)
        actor_states = torch.load(
            flags.actor_load_dir, map_location=flags.learner_device
        )
        model.load_state_dict(actor_states["model_state_dict"])
        optimizer.load_state_dict(actor_states["optimizer_state_dict"])
        scheduler.load_state_dict(actor_states["scheduler_state_dict"])

        if flags.use_rnd and "rnd_truth_net_state_dict" in actor_states:
            rnd_truth_net.load_state_dict(actor_states["rnd_truth_net_state_dict"])
            rnd_pred_net.load_state_dict(actor_states["rnd_pred_net_state_dict"])
            rnd_optimizer.load_state_dict(actor_states["rnd_optimizer"])
    
    if flags.pred_model and flags.pred_model_load_dir and os.path.exists(flags.pred_model_load_dir):
        logging.info("Loading pred_model: %s" % flags.pred_model_load_dir)
        pred_model_states = torch.load(
            flags.pred_model_load_dir, map_location=flags.learner_device
        )
        pred_model.load_state_dict(pred_model_states["pred_model_state_dict"])
        pred_optimizer.load_state_dict(pred_model_states["pred_optimizer_state_dict"])

    rew_classifier = create_rew_classifier(flags, learner_device)
    actor_rew_classifier = create_rew_classifier(flags, actor_device)


    if flags.no_train_actor:
        for g in optimizer.param_groups:
            g['lr'] = 0.0
            g['initial_lr'] = 0.0
        if rnd_optimizer is not None:
            for g in rnd_optimizer.param_groups:
                g['lr'] = 0.0
                g['initial_lr'] = 0.0

    stats = {}

    if flags.checkpoint and os.path.exists(flags.checkpoint):
        logging.info("Loading checkpoint: %s" % flags.checkpoint)
        checkpoint_states = torch.load(
            flags.checkpoint, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])

        if pred_model is not None and "pred_model_state_dict" in checkpoint_states:
            pred_model.load_state_dict(checkpoint_states["pred_model_state_dict"])
            pred_optimizer.load_state_dict(checkpoint_states["pred_optimizer_state_dict"])

        if flags.use_rnd and "rnd_truth_net_state_dict" in checkpoint_states:
            rnd_truth_net.load_state_dict(checkpoint_states["rnd_truth_net_state_dict"])
            rnd_pred_net.load_state_dict(checkpoint_states["rnd_pred_net_state_dict"])
            rnd_optimizer.load_state_dict(checkpoint_states["rnd_optimizer"])

        stats = checkpoint_states["stats"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    do_clustering = pred_model and flags.do_clustering
    if do_clustering:
        clustering_buffers = ([], [])
    else:
        clustering_buffers = None

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                learner_queue,
                model,
                actor_model,
                pred_model,
                optimizer,
                pred_optimizer,
                (rnd_truth_net, rnd_pred_net),
                rnd_optimizer,
                rew_classifier,
                clustering_buffers if i == 0 else None,
                scheduler,
                stats,
                flags,
                plogger,
                learner_device,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, actor_model, actor_rew_classifier, replay_buffer, flags, actor_device),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint(checkpoint_path=None):
        if flags.checkpoint:
            if checkpoint_path is None:
                checkpoint_path = flags.checkpoint
            logging.info("Saving checkpoint to %s", checkpoint_path)
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            }
            if pred_model is not None:
                save_dict.update({
                    "pred_model_state_dict": pred_model.state_dict(),
                    "pred_optimizer_state_dict": pred_optimizer.state_dict()
                })
            if rnd_truth_net is not None:
                save_dict.update({
                    "rnd_truth_net_state_dict": rnd_truth_net.state_dict(),
                    "rnd_pred_net_state_dict": rnd_pred_net.state_dict(),
                    "rnd_optimizer_state_dict": rnd_optimizer.state_dict(),
                })
            preemptive_save(save_dict, checkpoint_path)
            logging.info("Saving completed.")

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        train_start_time = timeit.default_timer()
        train_time_offset = stats.get("train_seconds", 0)  # used for resuming training
        last_checkpoint_time = timeit.default_timer()

        dev_checkpoint_intervals = [0, 0.25, 0.5, 0.75]

        loop_start_time = timeit.default_timer()
        loop_start_step = stats.get("step", 0)
        while True: 
            if do_clustering:
                if len(clustering_buffers[0]) >= flags.clustering_maxlen:
                    break
            loop_start_episode = stats.get("episode", 0)
            if flags.total_episodes is not None:
                if loop_start_episode >= flags.total_episodes:
                    break
            elif loop_start_step >= flags.total_steps:
                break
            time.sleep(5)
            loop_end_time = timeit.default_timer()
            loop_end_step = stats.get("step", 0)

            stats["train_seconds"] = round(
                loop_end_time - train_start_time + train_time_offset, 1
            )

            if loop_end_time - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = loop_end_time

            if len(dev_checkpoint_intervals) > 0:
                step_percentage = loop_end_step / flags.total_steps
                i = dev_checkpoint_intervals[0]
                if step_percentage > i:
                    checkpoint(flags.checkpoint[:-4] + "_" + str(i) + ".tar")
                    dev_checkpoint_intervals = dev_checkpoint_intervals[1:]

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                loop_end_step,
                (loop_end_step - loop_start_step) / (loop_end_time - loop_start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in sorted(stats.items())
                ),
            )
            loop_start_time = loop_end_time
            loop_start_step = loop_end_step
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])

    checkpoint()

    # Done with learning. Let's stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actorpool_thread.join()

    for t in learner_threads + inference_threads:
        t.join()

    results = {
        "savedir": flags.savedir
    }

    if do_clustering:
        logging.info("Collected enough data. Start clustering!")
        k = cluster(*clustering_buffers, save_dir=flags.savedir)
        results["num_clusters"] = k

    preemptive_save(results, flags.savedir + "/results.json", type="json")


def test(flags):
    test_checkpoint = os.path.join(flags.savedir, "test_checkpoint.tar")
    checkpoint = os.path.join(flags.load_dir, "checkpoint.tar")
    if not os.path.exists(os.path.dirname(test_checkpoint)):
        os.makedirs(os.path.dirname(test_checkpoint))

    logging.info("Creating test copy of checkpoint '%s'", checkpoint)

    checkpoint = torch.load(checkpoint)
    for d in checkpoint["optimizer_state_dict"]["param_groups"]:
        d["lr"] = 0.0
        d["initial_lr"] = 0.0

    checkpoint["scheduler_state_dict"]["last_epoch"] = 0
    checkpoint["scheduler_state_dict"]["_step_count"] = 0
    checkpoint["scheduler_state_dict"]["base_lrs"] = [0.0]
    checkpoint["stats"]["step"] = 0
    checkpoint["stats"]["_tick"] = 0

    flags.checkpoint = test_checkpoint
    flags.learning_rate = 0.0

    logging.info("Saving test checkpoint to %s", test_checkpoint)
    preemptive_save(checkpoint, test_checkpoint)
    logging.info("Saving completed.")

    train(flags)


def main(flags):
    if flags.wandb:
        wandb.init(
            project=flags.project,
            config=vars(flags),
            group=flags.group,
            entity=flags.entity,
            name=flags.wandb_name,
            id=flags.group + "-" + flags.wandb_name,
            resume=True
        )
    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)
    elif flags.mode.startswith("test"):
        test(flags)
    if flags.wandb:
        wandb.finish()
