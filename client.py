import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
from utils import init_distributed, get_local_dir, make_logger_path
import torch.multiprocessing as mp
import trainers
from typing import Optional, Set
import resource
import copy
import os
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter

class Client:

    def __init__(self, 
                 client_idx: int, 
                 local_train_data: dict, 
                 local_eval_data: dict, 
                 config: DictConfig,
                 TrainerClass, 
                 policy: nn.Module = None):
        
        self.client_idx = client_idx
        self.batch_counter = 0
        self.example_counter = 0
        self.round_counter = 0

        self.data = {"train": local_train_data, "test": local_eval_data}
        self.train_sample_num = len(local_train_data)

        self.config = config

        self.policy = policy
        self.TrainerClass = TrainerClass

        self.logger_dir = make_logger_path(f"Client-{self.client_idx}", config)
        
    def train(self, reference_model: Optional[nn.Module] = None):

        if 'FSDP' in self.config.trainer:
            world_size = torch.cuda.device_count()
            print('starting', world_size, 'processes for FSDP training')
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
            mp.spawn(self.worker_main,
                     nprocs=world_size,
                     args=(world_size, reference_model),
                     join=True)
        else:
            print('starting single-process worker')
            self.worker_main(0, 1, reference_model)        

        self.batch_counter += 1000
    
    def worker_main(self,
                    rank: int,
                    world_size: int,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        print(f'Creating trainer on process {rank} with world size {world_size}')

        trainer = self.TrainerClass(self.batch_counter,
                                    self.example_counter,
                                    self.logger_dir,
                                    self.client_idx,
                                    self.policy,
                                    self.config,
                                    self.config.seed,
                                    self.config.local_run_dir,
                                    dataset=self.data,
                                    reference_model=reference_model,
                                    rank=rank,
                                    world_size=world_size)
        trainer.train()
        trainer.save()

    def get_policy_params(self):
        return copy.deepcopy(self.policy.state_dict())
    
    def get_train_sample_num(self):
        return self.train_sample_num
