import copy
import logging
import random

import numpy as np
import torch
import wandb
import trainers

from client import Client
from agg import agg_FedAvg

from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import torch.nn as nn
import os
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
import resource
import torch.multiprocessing as mp


class FedAvgAPI(object):

    def __init__(self, local_train_data, global_train_data, local_test_data,
                 global_test_data, config, global_policy, local_policies,
                 reference_model) -> None:
        self.config = config
        self.train_data_global = global_train_data
        self.test_data_global = global_test_data

        self.data_global = {
            "train": global_train_data,
            "test": global_test_data
        }

        self.policy_global = global_policy

        self.client_list = []
        self.train_data_local = local_train_data
        self.test_data_local = local_test_data

        self._setup_clients(local_train_data, local_test_data, local_policies)

        self.reference_model = reference_model

    def _setup_clients(self, local_train_data, local_test_data,
                       local_policies):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.config.client_num_in_total):
            TrainerClass = getattr(trainers, self.config.trainer)
            c = Client(client_idx, local_train_data[client_idx],
                       local_test_data[client_idx], self.config, TrainerClass,
                       local_policies[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def train(self):
        for round_idx in range(self.config.comm_round):
            logging.info(
                "##################Communication round: {}".format(round_idx))

            w_locals = []

            for idx, client in enumerate(self.client_list):
                print("##########Client {} training".format(idx))
                client.train(self.reference_model)
                # we first suppose data is evenly distributed
                w_locals.append((1, copy.deepcopy(client.get_policy_params())))

            print("Aggregation begins")
            w_global = self._aggregate(w_locals)

            self.policy_global.load_state_dict(copy.deepcopy(w_global))

            for idx, client in enumerate(self.client_list):
                client.policy.load_state_dict(copy.deepcopy(w_global))

            if round_idx == self.config.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.config.frequency_of_the_test == 0:
                self._global_test(round_idx)

    def _global_test(self, round_idx):

        logging.info("################global_test : {}".format(round_idx))

        if 'FSDP' in self.config.trainer:
            world_size = torch.cuda.device_count()
            print('starting', world_size, 'processes for FSDP training')
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
            mp.spawn(self.worker_main,
                     nprocs=world_size,
                     args=(world_size, self.reference_model),
                     join=True)
        else:
            print('starting single-process worker')
            self.worker_main(0, 1, self.reference_model)

    def worker_main(self,
                    rank: int,
                    world_size: int,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        if self.config.debug:
            wandb.init = lambda *args, **kwargs: None
            wandb.log = lambda *args, **kwargs: None

        if rank == 0 and self.config.wandb.enabled:
            os.environ['WANDB_CACHE_DIR'] = get_local_dir(
                self.config.local_dirs)
            wandb.init(
                entity=self.config.wandb.entity,
                project=self.config.wandb.project,
                config=OmegaConf.to_container(self.config),
                dir=get_local_dir(self.config.local_dirs),
                name=self.config.exp_name,
            )

        print(
            f'Creating trainer on process {rank} with world size {world_size}')

        TrainerClass = getattr(trainers, self.config.trainer)
        trainer = TrainerClass(self.policy_global,
                                    self.config,
                                    self.config.seed,
                                    self.config.local_run_dir,
                                    dataset=self.data_global,
                                    reference_model=reference_model,
                                    rank=rank,
                                    world_size=world_size)

        trainer.test()
        trainer.save()
