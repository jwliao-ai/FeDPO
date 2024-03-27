import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
from utils import init_distributed, make_logger_path
import torch.multiprocessing as mp
import trainers
from typing import Optional
import resource
from omegaconf import DictConfig

class Server:
    def __init__(self,
                 global_train_data: dict, 
                 global_eval_data: dict, 
                 config: DictConfig, 
                 policy: nn.Module
                 ):
        
        self.server_idx = 999
        self.data = {"train": global_train_data, "test": global_eval_data}
        self.config = config
        self.policy = policy
        self.logger_dir = make_logger_path(f"Server", config)
        self.acc = 0.0
        
    def test(self, reference_model: Optional[nn.Module] = None):

        parent_conn, child_conn = mp.Pipe()
        
        if 'FSDP' in self.config.trainer:
            world_size = torch.cuda.device_count()
            print('starting', world_size, 'processes for FSDP training')
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
            mp.spawn(self.worker_main,
                     nprocs=world_size,
                     args=(world_size, child_conn, reference_model),
                     join=True)
        else:
            print('starting single-process worker')
            self.worker_main(0, 1, child_conn, reference_model)

        while parent_conn.poll():
            self.acc = parent_conn.recv()

    def worker_main(self,
                    rank: int,
                    world_size: int,
                    child_conn,
                    reference_model: Optional[nn.Module] = None
                    ):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        if 'FSDP' in self.config.trainer:
            init_distributed(rank, world_size, port=self.config.fsdp_port)

        print(f'Creating trainer on process {rank} with world size {world_size}')

        TrainerClass = getattr(trainers, self.config.trainer)
        trainer = TrainerClass(0,
                               0,
                               1.0,
                               self.logger_dir,
                               self.server_idx,
                               self.policy,
                               self.config,
                               self.config.seed,
                               self.config.local_run_dir,
                               dataset=self.data,
                               reference_model=reference_model,
                               rank=rank,
                               world_size=world_size)
        trainer.test()
        trainer.logger.close()
        if rank == 0: child_conn.send(trainer.eval_acc)

    def get_policy_params(self):
        return self.policy.parameters()

    def set_parameters(self, new_policy: nn.Module):
        for old_param, new_param in zip(self.policy.parameters(), new_policy.parameters()):
            old_param.data = new_param.data.clone()