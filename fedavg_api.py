import copy
import logging
import torch.nn as nn
import torch
import numpy as np

from client import Client
from server import Server
from utils import make_logger_path, init_distributed
from omegaconf import DictConfig
from typing import Optional, Set
from tensorboardX import SummaryWriter

class FedAvgAPI:

    def __init__(self, 
                 local_train_data: list[dict], 
                 global_train_data: dict, 
                 test_data: dict, 
                 config: DictConfig, 
                 policy: nn.Module,
                 reference_model: nn.Module):
    
        self.reference_model = reference_model
        self.comm_round = config.comm_round
        self.test_freq = config.frequency_of_test
        self.temp = config.temp

        self.policy_global = policy

        self.client_list = []
        self._setup_clients(local_train_data, test_data, config, policy)
        self._setup_server(global_train_data, test_data, config, policy)

        self.logger_dir = make_logger_path(f"Server", config)
        self.logger = SummaryWriter(self.logger_dir, flush_secs=1, max_queue=1)
        

    def _setup_clients(self, local_train_data: list[dict], test_data: dict, config: DictConfig, policy: nn.Module):
        print("-"*20 + " Setup clients (START) " + "-"*20)
        for client_idx in range(config.client_num_in_total):
            c = Client(client_idx, local_train_data[client_idx], test_data, config, copy.deepcopy(policy))
            self.client_list.append(c)
        print("-"*20 + " Setup clients (END) " + "-"*20)


    def _setup_server(self, global_train_data: dict, global_eval_data: dict, config: DictConfig, policy: nn.Module):
        print("-"*20 + " Setup server (START) " + "-"*20)
        self.server = Server(global_train_data, global_eval_data, config, copy.deepcopy(policy))
        print("-"*20 + " Setup server (END) " + "-"*20)


    def train(self):

        for round_idx in range(self.comm_round):

            logging.info("-"*62)
            logging.info("-"*20 + f" Communication Round {round_idx} " + "-"*20)
            logging.info("-"*62)
                        
            if round_idx == self.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.test_freq == 0:
                self._global_test(round_idx)
                
            w_locals = []
            ratios = []
            client_accs = []

            for client in self.client_list:
                assert type(client) == Client
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} training (START) " + "-"*20)
                print(f"Client {client.client_idx} has {client.train_sample_num} samples for traininig...")
                client.train(self.server.acc, self.reference_model)
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} training (END) " + "-"*20)
                logging.info("-"*20 + f" Round {round_idx}： Accuracy of client {client.client_idx}: {client.acc}" + "-"*20)
                w_locals.append(client.get_policy_params())
                client_accs.append(client.acc)
                ratios.append(np.exp(self.temp * client.acc))
                self.logger.add_scalar(f"acc/client-{client.client_idx}", client.acc, round_idx)

            self.logger.add_scalar(f"avg_acc/client", np.mean(client_accs), round_idx)
            print("-"*20 + f" Round {round_idx}: Aggregation (START) " + "-"*20)
            ratios = ratios / sum(ratios)
            logging.info("-"*20 + f" Round {round_idx}: Client ratios: {ratios} " + "-"*20)
            self.aggregate(w_locals, ratios)
            print("-"*20 + f" Round {round_idx}: Aggregation (END) " + "-"*20)
            self.send_parameters()


    def _global_test(self, round_idx):

        print("-"*20 + f" Round {round_idx}: Global test (START) " + "-"*20)

        self.server.test(self.reference_model)
        self.acc_global = self.server.acc
        
        logging.info("-"*20 + f" Round {round_idx}: Accuracy of server: {self.server.acc} " + "-"*20)
        print("-"*20 + f" Round {round_idx}: Global test (END) " + "-"*20)

        self.logger.add_scalar(f"avg_acc/server", self.server.acc, round_idx)
 

    def aggregate(self, w_locals, ratios: list):
        self.reset_global_policy()
        for i, w_local in enumerate(w_locals):
            for global_param, local_param in zip(self.policy_global.parameters(), w_local):
                global_param.data = global_param.data + local_param.data.clone() * ratios[i]
        self.server.set_parameters(self.policy_global)


    def send_parameters(self):
        for client in self.client_list:
            client.set_parameters(self.policy_global)


    def reset_global_policy(self):
        for param in self.policy_global.parameters():
            param.data = torch.zeros_like(param.data)