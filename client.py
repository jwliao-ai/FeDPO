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


class Client:

    def __init__(self, client_idx, local_train_data, local_eval_data, config,
                 TrainerClass, policy) -> None:
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_eval_data = local_eval_data

        self.config = config

        self.policy = policy
        self.TrainerClass = TrainerClass

    def train(self):
        trainer = self.TrainerClass(self.policy,
                                    self.config,
                                    self.config.seed,
                                    self.config.local_run_dir,
                                    reference_model=self.reference_model,
                                    rank=self.rank,
                                    world_size=self.world_size)
        trainer.train()
        trainer.save()
        return trainer.get_policy_params()
  
    def worker_main(self,
                    rank: int,
                    world_size: int,
                    config: DictConfig,
                    policy: nn.Module,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in config.trainer:
            init_distributed(rank, world_size, port=config.fsdp_port)

        if config.debug:
            wandb.init = lambda *args, **kwargs: None
            wandb.log = lambda *args, **kwargs: None

