import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
from utils import init_distributed, make_logger_path
import torch.multiprocessing as mp
import trainers
from typing import Optional, Set
import resource
from omegaconf import OmegaConf, DictConfig

class Client:

    def __init__(self, 
                 client_idx: int, 
                 local_train_data: dict, 
                 local_eval_data: dict, 
                 config: DictConfig,
                 policy: nn.Module = None):
        
        self.client_idx = client_idx
        self.batch_counter = 0
        self.example_counter = 0
        self.decay = 1.0

        self.data = {"train": local_train_data, "test": local_eval_data}
        self.train_sample_num = len(local_train_data)
        self.config = config
        self.policy = policy
        self.logger_dir = make_logger_path(f"Client-{self.client_idx}", config)
        self.eval_acc = 0.
        self.device = torch.device("cuda")
        
    def test(self, server_acc, reference_model: Optional[nn.Module] = None):

        trainer = trainers.BasicTrainer(self.batch_counter,
                                        self.example_counter,
                                        self.decay,
                                        self.logger_dir,
                                        self.client_idx,
                                        self.policy,
                                        self.config,
                                        self.config.seed,
                                        self.config.local_run_dir,
                                        dataset=self.data,
                                        reference_model=reference_model,
                                        rank=0,
                                        world_size=1)
        self.eval_acc = trainer.test()
        if self.eval_acc <= server_acc:
            self.decay *= self.config.decay_rate

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

        self.batch_counter += 310
        self.example_counter += 19840
    
    def worker_main(self,
                    rank: int,
                    world_size: int,
                    reference_model: Optional[nn.Module] = None):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        print(f'Creating trainer on process {rank} with world size {world_size}')

        TrainerClass = getattr(trainers, self.config.trainer)
        trainer = TrainerClass(self.batch_counter,
                               self.example_counter,
                               self.decay,
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
        for param in self.policy.parameters():
            param.detach()
        return self.policy.parameters()
    
    def get_train_sample_num(self):
        return self.train_sample_num

    def set_parameters(self, policy_global):
        for old_param, new_param in zip(self.policy.parameters(), policy_global.parameters()):
            old_param.data = new_param.data.clone().to(self.device)