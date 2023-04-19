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

import collections
import torch
from torch import nn
from torch.nn import functional as F


from ..env import get_env
import numpy as np


def get_torso(flags):
    env = get_env(flags)
    return AtariTorso


class BaseNet(nn.Module):
    """This base class simply provides a skeleton for running with torchbeast."""

    AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")
    ExtendedAgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline objective")

    def __init__(self, flags):
        super(BaseNet, self).__init__()
        self.num_rewards = flags.num_objectives + 1 if flags.multi_objective else 1
        self.register_buffer("reward_sum", torch.zeros((self.num_rewards,)))
        self.register_buffer("reward_m2", torch.zeros((self.num_rewards,)))
        self.register_buffer("reward_count", torch.zeros((self.num_rewards,)).fill_(1e-8))

    def forward(self, inputs, core_state):
        raise NotImplementedError

    def initial_state(self, batch_size=1):
        return ()

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        batch_shape = reward_batch.shape[:-1]
        n_batch_dim = len(batch_shape)
        batch_dim = tuple(range(n_batch_dim))
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch, batch_dim)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2, batch_dim) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        # print(new_count, new_m2, new_sum)

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        # print(self.reward_m2, self.reward_count, torch.sqrt(self.reward_m2 / self.reward_count))
        return torch.sqrt(self.reward_m2 / self.reward_count) + 1e-8


class RNDNet(BaseNet):
    """
    This model was based on 'neurips2020release' tag on the NLE repo, itself
    based on Kuttler et al, 2020
    The NetHack Learning Environment
    https://arxiv.org/abs/2006.13760
    """

    def __init__(self, observation_space, action_space, flags, device, logits_mask=None):
        super(RNDNet, self).__init__(flags)

        self.flags = flags

        self.torso = get_torso(flags)(observation_space, action_space, flags, device)

        self.output_dim = flags.rnd_output_dim
        self.net = nn.Linear(flags.hidden_dim, self.output_dim)

        self.logits_mask = logits_mask is not None
        if self.logits_mask:
            self.policy_logits_mask = nn.parameter.Parameter(
                logits_mask, requires_grad=False
            )

    def forward(self, inputs):
        T, B = inputs["done"].shape

        st = self.torso(inputs).view(T, B, -1)

        # return self.net(st) / np.sqrt(self.output_dim)
        return self.net(st)


class BaselineNet(BaseNet):
    """
    This model was based on 'neurips2020release' tag on the NLE repo, itself
    based on Kuttler et al, 2020
    The NetHack Learning Environment
    https://arxiv.org/abs/2006.13760
    """

    def __init__(self, observation_space, action_space, flags, device, logits_mask=None):
        super(BaselineNet, self).__init__(flags)

        self.flags = flags

        self.num_actions = action_space.n
        self.use_lstm = flags.use_lstm
        self.h_dim = flags.hidden_dim

        self.torso = get_torso(flags)(observation_space, action_space, flags, device)

        old_version = "objective_as_input" not in flags

        self.multi_objective = flags.get("multi_objective", flags.num_objectives > 1)

        self.encode_completed_objectives = old_version and self.multi_objective and "completed" in observation_space.spaces
        if self.encode_completed_objectives:
            print("Encoding completed objectives!")
            self.completed_objectives_encoder = nn.Sequential(
                nn.Linear(flags.num_objectives + self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
            )
        
        self.objective_as_input = flags.get("objective_as_input", False) and self.multi_objective
        self.num_objectives = flags.num_objectives if old_version else flags.num_objectives + 1  # only used when self.multi_objective

        if self.objective_as_input:
            print("Objective as input!!", flush=True)
            self.objective_encoder = nn.Sequential(
                nn.Linear(self.num_objectives + self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
            )

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        self.num_policies = 1 if self.objective_as_input or not self.multi_objective else self.num_objectives
        self.policy = nn.Linear(self.h_dim, self.num_actions * self.num_policies)
        self.baseline = nn.Linear(self.h_dim, self.num_policies)

        self.logits_mask = logits_mask is not None
        if self.logits_mask:
            self.policy_logits_mask = nn.parameter.Parameter(
                logits_mask, requires_grad=False
            )
        

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state, learning=False):
        T, B = inputs["done"].shape

        st = self.torso(inputs)

        if self.encode_completed_objectives:
            st = self.completed_objectives_encoder(torch.cat((st, inputs["completed"].view(T * B, -1)), 1))
        
        if self.objective_as_input:
            st = self.objective_encoder(torch.cat((st, F.one_hot(inputs["objective"].view(T * B).to(torch.int64), self.num_objectives)), 1))

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * t for t in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B' x A]
        policy_logits = self.policy(core_output).view(T * B * self.num_policies, -1)

        # -- [B' x 1]
        baseline = self.baseline(core_output)

        if self.logits_mask:
            policy_logits = policy_logits * self.policy_logits_mask + (
                (1 - self.policy_logits_mask) * -1e10
            )

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        if self.objective_as_input or not self.multi_objective:
            policy_logits = policy_logits.view(T, B, -1)
            baseline = baseline.view(T, B)
            action = action.view(T, B)
        else:
            policy_logits = policy_logits.view(T, B, self.num_policies, -1)
            baseline = baseline.view(T, B, self.num_policies)
            action = action.view(T, B, self.num_policies)

        output = dict(policy_logits=policy_logits, baseline=baseline, action=action, core_output=core_output.view(T, B, -1), obs_st=st.view(T, B, -1))
        return (output, core_state)


class PredictionModel(nn.Module):
    def __init__(self, observation_space, action_space, flags, device, predict_items={"reward": ()}, version=1):
        super(PredictionModel, self).__init__()

        self.flags = flags
        self.predict_items = predict_items
        self.version = version

        self.num_actions = action_space.n
        self.h_dim = flags.hidden_dim
        
        env = get_env(flags)
        if env == 'nethack':
            self.obs_keys = ("glyphs", "message", "blstats", "chars", "colors", "specials")
        else:
            self.obs_keys = ("frame",)

        self.torso = get_torso(flags)(observation_space, action_space, flags, device)

        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + self.num_actions + 1, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        if version == 1:
            self.reward_prediction = nn.Linear(self.h_dim, 1)
        elif version == 2:
            predict_heads = {}
            if "reward" in predict_items:
                predict_heads["reward"] = nn.Linear(self.h_dim, 1)
            
            assert len(predict_heads) > 0
            self.predict_heads = nn.ModuleDict(predict_heads)

    def forward(self, inputs, action, done, last_obs=None, training=True):
        obs_keys = self.obs_keys
        T, B, *_ = action.shape
        if last_obs is None:
            last_obs = dict()
            for key in obs_keys:
                t = torch.zeros_like(inputs[key])
                t[1:].copy_(inputs[key][:-1])
                last_obs[key] = t
        # print(inputs["glyphs"].shape, last_obs["glyphs"].shape, inputs["blstats"].shape, last_obs["blstats"].shape)

        if ("frame" in self.predict_items and not self.version == 3) or self.flags.get("pred_no_next_frame", False):
            next_obs_mask = 0
        else:
            next_obs_mask = 1
        new_obs = { key: torch.cat((inputs[key] * next_obs_mask, last_obs[key]), dim=1) for key in obs_keys }

        obs_st = self.torso(new_obs)

        obs_st = obs_st.view(T, B * 2, -1)
        hh_dim = self.h_dim // 2
        obs_st = torch.cat(((1 - done.float().unsqueeze(-1)) * obs_st[:, :B, :hh_dim], obs_st[:, B:, hh_dim:]), dim=2)
        obs_st = obs_st.view(T * B, -1)

        # -- [B' x num_actions]
        one_hot_action = F.one_hot(
            action.view(T * B), self.num_actions
        ).float()

        st = torch.cat((obs_st, one_hot_action, done.float().view(T * B, 1)), 1)

        st = self.fc(st) + obs_st

        predictions = dict()
        if self.version == 1:
            predictions["reward"] = self.reward_prediction(st).view(T, B)
        else:
            for key, head in self.predict_heads.items():
                predictions[key] = head(st).view(T, B, *self.predict_items[key])

        return (
            predictions,
            {},
            obs_st.clone().view(T, B, -1),
            st.clone().view(T, B, -1)
        )


class RewardClassifier(nn.Module):

    def __init__(self, pred_net: PredictionModel, centroids: np.ndarray, threshold: float, device, cluster_names=None, noise=0.0):
        super(RewardClassifier, self).__init__()

        self.pred_net = pred_net
        self.num_clusters = centroids.shape[0]
        self.centroids = torch.from_numpy(centroids).to(device=device, dtype=torch.float)[None, None, :, :]  # [1, 1, K, D]
        self.threshold = threshold
        self.noise = noise
        self.cluster_names = cluster_names + [f"{self.num_clusters:2}:new_tasks"]

    def forward(self, inputs, action, done, last_obs=None, one_hot=False, return_embeds=False):
        embeds = self.pred_net(inputs, action, done, last_obs=last_obs, training=False)[3].detach()
        embeds = embeds + torch.randn_like(embeds) * self.noise
        dis = (embeds[:, :, None, :] - self.centroids).square().sum(-1)  # [T, B, K]
        min_dis = torch.min(dis, dim=-1)
        classes = torch.where(min_dis.values < self.threshold, min_dis.indices, self.num_clusters)
        if one_hot:
            result = F.one_hot(classes, self.num_clusters + 1)
        else:
            result = classes
        if return_embeds:
            return result, embeds
        else:
            return result


class AtariTorso(nn.Module):
    def __init__(self, observation_space, action_space, flags, device):
        super(AtariTorso, self).__init__()
        self.observation_shape = observation_space['frame'].shape
        self.num_actions = action_space.n

        self.h_dim = flags.hidden_dim

        # Feature extraction.
        # [3, 84, 84]
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )  
        # [32, 20, 20]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # [64, 9, 9]
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # [64, 7, 7]

        out_dim = 3136

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = self.fc(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
