import logging


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
