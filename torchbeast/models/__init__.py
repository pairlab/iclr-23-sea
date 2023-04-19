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

from platform import architecture

from omegaconf import OmegaConf
import torch
import numpy as np
import joblib


from .baseline import BaselineNet, PredictionModel, RewardClassifier, RNDNet
from ..env import make_env, get_env


def create_model(flags, device):
    model_string = flags.model
    if model_string == "baseline":
        model_cls = BaselineNet
    else:
        raise NotImplementedError("model=%s" % model_string)

    env = make_env(flags.env, flags=flags, dummy_env=True)
    logits_mask = None

    model = model_cls(env.observation_space, env.action_space, flags, device, logits_mask=logits_mask)
    model.to(device=device)
    return model


def create_pred_model(flags, device):
    env = make_env(flags.env, flags=flags, dummy_env=True)

    predict_items = flags.get("predict_items", None)
    if predict_items is None:
        model = PredictionModel(env.observation_space, env.action_space, flags, device)
    else:
        model = PredictionModel(env.observation_space, env.action_space, flags, device, predict_items=predict_items, version=2)
    model.to(device=device)
    return model


def create_rnd_net(flags, device):
    env = make_env(flags.env, flags=flags, dummy_env=True)
    logits_mask = None

    truth_net = RNDNet(env.observation_space, env.action_space, flags, device, logits_mask=logits_mask)
    pred_net = RNDNet(env.observation_space, env.action_space, flags, device, logits_mask=logits_mask)
    truth_net.to(device=device)
    pred_net.to(device=device)
    return truth_net, pred_net


def create_rew_classifier(flags, device):
    # print(device)
    if flags.cluster_load_dir is None or flags.cluster_pred_model_load_dir is None:
        return None
    cluster_data = joblib.load(flags.cluster_load_dir)
    if type(cluster_data) == dict:
        cluster_names = cluster_data["names"]
        centroids = cluster_data["centroids"]
        threshold = cluster_data["threshold"]
    else:
        cluster_names, centroids = cluster_data
        threshold = flags.cluster_threshold
    assert len(cluster_names) == centroids.shape[0] == flags.num_objectives
    cluster_pred_model = load_pred_model(flags.cluster_pred_model_load_dir, device)
    cluster_pred_model.to(device=device)
    print("Creating RewardClassifier ...")
    print("cluster_load_dir:", flags.cluster_load_dir)
    print("cluster_pred_model_load_dir:", flags.cluster_pred_model_load_dir)
    rew_classifier = RewardClassifier(cluster_pred_model, centroids, threshold, device, cluster_names, noise=flags.cluster_noise)
    rew_classifier.to(device=device)
    return rew_classifier


def load_model(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_model(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model


def load_pred_model(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_pred_model(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    if "pred_model_state_dict" in checkpoint_states:
        model.load_state_dict(checkpoint_states["pred_model_state_dict"])
        return model
    else:
        return None


def load_rnd_net(load_dir, device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    truth_net, pred_net = create_rnd_net(flags, device)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    truth_net.load_state_dict(checkpoint_states["rnd_truth_net_state_dict"])
    pred_net.load_state_dict(checkpoint_states["rnd_pred_net_state_dict"])
    return truth_net, pred_net
