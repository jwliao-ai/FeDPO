import copy
import logging
import resource
import torch.multiprocessing as mp
import torch.nn as nn
import torch
import trainers
import numpy as np

from client import Client
from utils import make_logger_path, init_distributed
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from tensorboardX import SummaryWriter

class FedAvgAPI(object):

    def __init__(self, 
                 local_train_data: list[dict], 
                 global_train_data: dict, 
                 test_data: dict, 
                 config: DictConfig, 
                 policy: nn.Module,
                 reference_model: nn.Module):
        
        self.global_batch_counter = 0
        self.global_example_counter = 0
        self.config = config

        self.test_data = test_data
        self.reference_model = reference_model

        self.data_global = {
            "train": global_train_data,
            "test": test_data
        }

        self.policy_global = copy.deepcopy(policy)
        self.acc_global = 0.

        self.client_list = []
        self.train_data_local = local_train_data
        self._setup_clients(local_train_data, copy.deepcopy(policy))

        self.logger_dir = make_logger_path(f"Server", self.config)
        self.logger = SummaryWriter(self.logger_dir, flush_secs=1, max_queue=1)
        
    def _setup_clients(self, local_train_data: list[dict], policy: nn.Module):

        print("-"*20 + " Setup clients (START) " + "-"*20)
        for client_idx in range(self.config.client_num_in_total):
            c = Client(client_idx, local_train_data[client_idx], self.test_data, self.config, copy.deepcopy(policy))
            self.client_list.append(c)
        print("-"*20 + " Setup clients (END) " + "-"*20)

    def train(self):
        for round_idx in range(self.config.comm_round):
            logging.info("-"*64)
            logging.info("-"*20 + f" Communication Round: {round_idx} " + "-"*20)
            logging.info("-"*64)
            
            if round_idx == self.config.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.config.frequency_of_the_test == 0:
                self._global_test(round_idx)
                
            w_locals = []
            ratios = []
            client_accs = []

            for client in self.client_list:
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} training (START) " + "-"*20)
                print(f"client {client.client_idx} has {client.train_sample_num} samples for traininig...")
                client.train(self.reference_model)
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} training (END) " + "-"*20)
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} testing (START) " + "-"*20)
                client.test(self.acc_global, self.reference_model)
                logging.info("-"*20 + f" Round {round_idx}, Accuracy of Client-{client.client_idx}: {client.eval_acc}.")
                print("-"*20 + f" Round {round_idx}: Client {client.client_idx} testing (END) " + "-"*20)
                w_locals.append(client.get_policy_params())
                client_accs.append(client.eval_acc)
                ratios.append(np.exp(self.config.temp_a * client.eval_acc))
                self.logger.add_scalar(f"acc/client-{client.client_idx}", client.eval_acc, round_idx)

            self.logger.add_scalar(f"avg_acc/client", np.mean(client_accs), round_idx)
            print("-"*20 + f" Round: {round_idx} Aggregation (START) " + "-"*20)
            ratios = ratios / sum(ratios)
            self.aggregate(w_locals, ratios)
            print("-"*20 + f" Round: {round_idx} Aggregation (END) " + "-"*20)
            self.send_parameters()

    def _global_test(self, round_idx):

        print("-"*20 + f" Round: {round_idx} Global test (START) " + "-"*20)

        parent_conn, child_conn = mp.Pipe()
        
        if 'FSDP' in self.config.trainer:
            world_size = torch.cuda.device_count()
            print('starting', world_size, 'processes for FSDP training')
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
            mp.spawn(self.worker_main,
                     nprocs=world_size,
                     args=(world_size, child_conn, self.reference_model),
                     join=True)
        else:
            print('starting single-process worker')
            self.worker_main(0, 1, child_conn, self.reference_model)

        while parent_conn.poll():
            self.acc_global = parent_conn.recv()
        
        logging.info("-"*20 + f" Round: {round_idx}: Accuracy of Server is: {self.acc_global} " + "-"*20)
        print("-"*20 + f" Round: {round_idx}: Global test (END) " + "-"*20)

        self.logger.add_scalar(f"avg_acc/server", self.acc_global, round_idx)
        
    def worker_main(self,
                    rank: int,
                    world_size: int,
                    child_conn,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        print(f'Creating trainer on process {rank} with world size {world_size}')

        TrainerClass = getattr(trainers, self.config.trainer)
        trainer = TrainerClass(self.global_batch_counter,
                               self.global_example_counter,
                               1.,
                               self.logger_dir,
                               999,
                               self.policy_global,
                               self.config,
                               self.config.seed,
                               self.config.local_run_dir,
                               dataset=self.data_global,
                               reference_model=reference_model,
                               rank=rank,
                               world_size=world_size)
        
        if self.config.server_train:
            trainer.train()
            trainer.save()
        else:
            trainer.test()
            if rank == 0: child_conn.send(trainer.eval_acc)

    def aggregate(self, w_locals, ratios):
        for param in self.policy_global.parameters():
            param.data = torch.zeros_like(param.data)
        for i, w_local in enumerate(w_locals):
            for policy_global_param, policy_local_param in zip(self.policy_global.parameters(), w_local):
                policy_global_param.data = policy_global_param.data + policy_local_param.data.clone() * ratios[i]

    def send_parameters(self):
        for client in self.client_list:
            client.set_parameters(self.policy_global)