# FeDPO: Federated Direct Preference Optimization

This codebase is based on the available git repository [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell). And some parts of this `README.md` draws from this repository.

The Python packages required to run FeDPO are listed in `requirements.txt`.

The experiments are performed on Python 3.10 in a Ubuntu 22.04 environment.

## Before running FeDPO

For general RLHF, the SFT stage essentially ensures that the preference data we train on is in-distribution for our policy before we actually do the learning from preferences part.

Before running FeDPO, you should have had a SFT pretrained model which you can get from running SFT of the codebase [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell).

## Running FeDPO

To run FeDPO, use the command like (on a custom model GPT2):

    python -u main.py model=blank_model model.name_or_path=../autodl-tmp/models/gpt2 model.block_name=GPT2Block datasets=[hh] loss=dpo loss.beta=0.1 temp=0.1 decay_rate=0.995 exp_name=openai_fedpo_gpt2 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer model.fsdp_policy_mp=bfloat16 model.archive=/path/to/checkpoint/from/sft/LATEST/policy.pt eval_every=6000 client_num_in_total=10 comm_round=100 frequency_of_test=1

> Note: `eval_every` is measured in **examples** and for **each client**.

On 4 24GB RTX 4090s, FeDPO training took about 5hrs to completely converge after 20 communication rounds.

The tensorboard logs will be saved in a folder `data/` and you can see the learning curves through the command:

    tensorboard --logdir data/

> ps: Though in my code, I used `TensorboardX` to plot learning curves, using `TensorboardX` is not a good choice for multi-GPU training (you need to remember to `.close()` the SummaryWriter). So you can use `wandb` instead for better logging experience.

### Customizing training
The options for training are in `config/config.yaml`, `config/model/blank_model.yaml`, and `config/loss/dpo.yaml`. See the comments in these files for more information on what they do.

You can use one of the pre-configured models by passing `model=some_model`, where `config/model/some_model.yaml` exists. We have a few examples already given.

If you want to use another model, just create a new config for that model (following our examples; it must be a `.yaml` file!), or use `model=blank_model` with `model.name_or_path=NAME_OR_PATH`, optionally `model.tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH` if it is different than the model's name/path, and `model.block_name=NAME_OF_TRANSFORMER_BLOCK` (if you are using FSDP). The only other options you might want to change are the dpo loss options, which are `loss.beta` and `loss.reference_free` (see `config/loss/dpo.yaml`).

## Trainer classes

Three different trainer classes are provided in `trainers.py`:
- `BasicTrainer`: For multiple GPUs, naively partition the model among them. e.g., for two GPUs, the first half of the model layers will be on GPU 0, the second half will be on GPU 1. This trainer effectively increases your available GPU memory without using multiple GPUs are once for compute (so you get no speedup).
- `FSDPTrainer`: Use PyTorch's [Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP) implementation to shard each transformer block amongst available GPUs. Should give a significant speedup over `BasicTrainer` with batch size per GPU >1. The batch size per gpu is equal to `batch_size / (gradient_accumulation_steps * num_gpus)`. **You may need to run `ulimit -n 64000` in your launch script before calling `train.py` with this trainer; e.g., `ulimit -n 64000; python train.py ...`.**
- `TensorParallelTrainer`: Use PyTorch tensor parallelism (with [this wrapper](https://github.com/BlackSamorez/tensor_parallel)) to shard each linear layer amongst available GPUs. This trainer is experimental, but should work.

### Which trainer do I use?
 For single GPU training, use `BasicTrainer`. For many-GPU setups, `FSDPTrainer` will most likely be the best choice, though these haven't been benchmarked yet.

# Adding new datasets
Adding new/custom datasets is easy, and shouldn't take more than 10 minutes or so. Add your dataset to `load_data.py` (Eric has implemented Anthropic-HH, Stanford Human Preferences, and StackExchange as references). Follow our reference datasets (in the functions `get_se()`, `get_shp()`, `get_hh()`); you essentially need to return a dict mapping each prompt to another dict containing four values:

- `responses: List[str]`: the list of responses on which preferences are given
- `pairs: List[Tuple[int]]`: the preference pairs, where the first value in each tuple is the preferred response and the second value is the dispreferred response
- `sft_target: str`: the response to use for this prompt during SFT (this response may or may not be one of the values in `responses`)
- `truncation_mode: str`: the mode of truncation to be applied during the task. It is either `keep_start` or `keep_end`, determining how the response is truncated when the combined length of the responses exceeds a certain limit.

# Tips for faster training on multiple GPUs
FSDP is recommended for faster training when multiple GPUs are available. In general, you should try to use a batch size of at least 2 on each GPU (i.e., `batch_size // (grad_accumulation_steps * N_GPUS)` is at least 2) to see a speedup from FSDP compared to the `BasicTrainer`. One way to do this is to use mixed precision. This repo implements mixed precision through [FSDP](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision). Enable mixed precision (only supported for `FSDPTrainer`, currently) by passing `model.fsdp_policy_mp=bfloat16` or `model.fsdp_policy_mp=float16` (only `bfloat16` has been tested). Another way to reduce memory usage is activation checkpointing (or *gradient checkpointing*), which can be enabled with `activation_checkpointing=true` (also implemented only for `FSDPTrainer`). Activation checkpointing doesn't always increase throughput, but if you're stuck at batch size per GPU of 1, it's worth a try.

See [this article](https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/) for more information about optimizing FSDP.

