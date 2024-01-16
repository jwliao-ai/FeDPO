import logging


class Client:

    def __init__(self, client_idx, local_train_data, local_eval_data, config,
                 device, model_trainer, policy) -> None:
        self.client_idx = client_idx
        self.local_train_data = local_train_data
        self.local_eval_data = local_eval_data

        self.config = config
        self.device = device

        self.policy = policy
        self.model_trainer = model_trainer

    def train(self):
        self.model_trainer.train()
