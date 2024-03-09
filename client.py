import logging
import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
import copy


class Client:

    def __init__(self, client_idx, local_train_data, local_eval_data, config,
                 TrainerClass, policy) -> None:
        self.client_idx = client_idx

        self.data = {"train": local_train_data, "test": local_eval_data}

        self.config = config

        self.policy = policy
        self.TrainerClass = TrainerClass

    def train(self, reference_model: Optional[nn.Module] = None):
        print(self.policy.state_dict())
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
        print(self.policy.state_dict())

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

        trainer = self.TrainerClass(self.policy,
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
