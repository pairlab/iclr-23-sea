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

import multiprocessing as mp
import logging
import os
import sys
import threading
import time
import hashlib

import torch

import libtorchbeast

from .env import get_env, make_env


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


# Helper functions for NethackEnv.
def _format_observation(obs):
    obs = torch.from_numpy(obs)
    return obs.view((1, 1) + obs.shape)  # (...) -> (T,B,...).


def create_folders(flags):
    # Creates some of the folders that would be created by the filewriter.
    logdir = os.path.join(flags.savedir, "archives")
    if not os.path.exists(logdir):
        logging.info("Creating archive directory: %s" % logdir)
        os.makedirs(logdir, exist_ok=True)
    else:
        logging.info("Found archive directory: %s" % logdir)


def create_env(flags, env_id=0, lock=threading.Lock(), dummy_env=False):

    with lock:
        env = get_env(flags)
        kwargs = { "env_id": env_id }
        env = make_env(flags.env, kwargs, flags, dummy_env=dummy_env)
        seed = abs(hash(f"{flags.seed}-env-{env_id}")) % (2**32)
        # print(seed)
        env.seed(seed)
        if flags.seedspath is not None and len(flags.seedspath) > 0:
            raise NotImplementedError("seedspath > 0 not implemented yet.")

        return env


def serve(flags, server_address, env_id):
    env = lambda: create_env(flags, env_id)
    server = libtorchbeast.Server(
        env, 
        server_address=server_address, 
        server_id=env_id, 
        done_at_reward=flags.done_at_reward,
        num_labels=flags.num_objectives + (1 if flags.include_new_tasks else 0)
    )
    server.run()


def main(flags):
    create_folders(flags)

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=serve, args=(flags, f"{flags.pipes_basename}.{i}", i), daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
