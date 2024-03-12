import os

import torch
import copy

import torch

torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, get_open_port
from load_data import get_dataset
from fedavg_api import FedAvgAPI
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import json
import socket
from typing import Optional, Set
import resource

OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to',
              config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {
        'device_map': 'balanced'
    } if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}'
        )
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    local_train_data, global_train_data = get_dataset(
        config.datasets[0],
        split='train',
        silent=False,
        cache_dir=get_local_dir(config.local_dirs),
        client_num_in_total=config.client_num_in_total)

    local_test_data, global_test_data = get_dataset(
        config.datasets[0],
        split='test',
        silent=False,
        cache_dir=get_local_dir(config.local_dirs),
        client_num_in_total=config.client_num_in_total)

    global_policy = copy.deepcopy(policy)
    local_policies = [
        copy.deepcopy(policy) for _ in range(config.client_num_in_total)
    ]

    fedavgAPI = FedAvgAPI(local_train_data, global_train_data, local_test_data,
                          global_test_data, config, global_policy,
                          local_policies, reference_model)

    fedavgAPI.train()

if __name__ == "__main__":
    main()