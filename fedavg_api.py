import copy
import logging
import random

import numpy as np
import torch
import wandb
import trainers

from client import Client
from agg import agg_FedAvg


class FedAvgAPI(object):

    def __init__(self, local_train_data, global_train_data, local_test_data,
                 global_test_data, config, global_policy, local_policies,
                 reference_model) -> None:
        self.config = config
        self.train_data_global = global_train_data
        self.test_data_global = global_test_data

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
            trainer = TrainerClass(local_policies[client_idx],
                                   self.config,
                                   self.config.seed,
                                   self.config.local_run_dir,
                                   reference_model=self.reference_model,
                                   rank=self.rank,
                                   world_size=self.world_size)
            c = Client(client_idx, local_train_data[client_idx],
                       local_test_data[client_idx], self.config, self.device,
                       trainer, local_policies[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def _global_test(self, round_idx):

        logging.info("################global_test : {}".format(round_idx))

    def train(self):
        w_global = self.model_trainer.getmo
