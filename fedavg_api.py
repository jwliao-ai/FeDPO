import copy
import logging
import resource
import torch.multiprocessing as mp
import torch.nn as nn
import torch
import trainers
import os

from client import Client
from agg import agg_FedAvg
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, make_logger_path
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

        self.data_global = {
            "train": global_train_data,
            "test": test_data
        }

        self.policy_global = copy.deepcopy(policy)

        self.client_list = []
        self.train_data_local = local_train_data

        self._setup_clients(local_train_data, copy.deepcopy(policy))

        self.reference_model = reference_model

        self.logger_dir = make_logger_path(f"Server", self.config)
        
    def _setup_clients(self, local_train_data, policy):
        logging.info("#"*20 + " Setup clients (START) " + "#"*20)
        for client_idx in range(self.config.client_num_in_total):
            TrainerClass = getattr(trainers, self.config.trainer)
            c = Client(client_idx, local_train_data[client_idx],
                       self.test_data, self.config, TrainerClass,
                       copy.deepcopy(policy))
            self.client_list.append(c)
        logging.info("#"*20 + " Setup clients (END) " + "#"*20)

    def train(self):

        for round_idx in range(self.config.comm_round):

            logging.info("#"*20 + f" Communication round: {round_idx} " + "#"*20)

            w_locals = []

            for idx, client in enumerate(self.client_list):
                logging.info("#"*20 + f" Client {idx} training (START) " + "#"*20)
                print(f"client {idx} has {client.train_sample_num} samples for traininig...")
                client.train(self.reference_model)
                logging.info("#"*20 + f" Client {idx} training (END) " + "#"*20)
                w_locals.append((client.get_train_sample_num(), copy.deepcopy(client.get_policy_params())))

            print("#"*20 + f" Start aggregation round: {round_idx} " + "#"*20)
            w_global = self._aggregate(w_locals)

            del w_locals
            
            self.policy_global.load_state_dict(copy.deepcopy(w_global))

            if round_idx == self.config.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.config.frequency_of_the_test == 0:
                self._global_test(round_idx)

            for idx, client in enumerate(self.client_list):
                client.policy.load_state_dict(copy.deepcopy(self.policy_global.state_dict()))

    def _global_test(self, round_idx):

        logging.info("#"*20 + f" global_test : {round_idx} " + "#"*20)

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

        self.global_batch_counter += 1000
        
    def worker_main(self,
                    rank: int,
                    world_size: int,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        print(
            f'Creating trainer on process {rank} with world size {world_size}')

        TrainerClass = getattr(trainers, self.config.trainer)
        trainer = TrainerClass(self.global_batch_counter,
                               self.global_example_counter,
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
        
        logging.info("#"*20 + f" Server has {len(self.data_global['train'])} samples for training and {len(self.data_global['test'])} samples for testing " + "#"*20)
        
        if self.config.server_train:
            trainer.train()
        else:
            trainer.test()
        trainer.save()

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)
