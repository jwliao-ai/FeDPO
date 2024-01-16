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
                w = client.train()
                # we first suppose data is evenly distributed
                w_locals.append((1, copy.deepcopy(w)))

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
