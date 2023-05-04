import torch
import numpy as np

import os, pickle

from contextlib import nullcontext

from data.gen_matrix_progs import gen_matrix_progs_col_stripe as gen_progs
from nets.nash_predictor_model import NashPredictorConv, NashPredictorValueConv

from utils.utils import initialize_code2vec_dataset


class BaseTrainer:
    """
    Basic supervised learning training/evaluation loops in Pytorch.
    """
    def __init__(self,
                 model,
                 device,
                 optimizer,
                 train_dataloader,
                 valid_dataloader,
                 config
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config

        self.output_dir = self.config['output_dir']
        os.system('mkdir -p ' + self.output_dir)

        model.to(device)

    def prepare_data(self, data):
        raise NotImplementedError

    def loss_function(self, model_outputs, ground_truth):
        raise NotImplementedError

    def learning_step(self, data, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        data_device = []
        for tensor in data:
            data_device.append(tensor.to(self.device))
        with torch.no_grad() if mode != 'train' else nullcontext():
            inputs, ground_truth = self.prepare_data(data_device)
            model_outputs = self.model(*inputs)
            loss = self.loss_function(model_outputs, ground_truth)

        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, tuple(x.cpu() for x in model_outputs)

    def train_iter(self, data):
        train_loss, _ = self.learning_step(data)
        return train_loss.cpu().item()

    def eval(self):
        valid_errors = []
        all_model_outputs = []
        for j, valid_data in enumerate(self.valid_dataloader, 0):
            valid_error, model_outputs = self.learning_step(valid_data, mode="eval")
            valid_errors.append(valid_error.cpu().item())
            all_model_outputs.append(model_outputs)

        return valid_errors, all_model_outputs

    def train_loop(self, num_epochs):
        for epoch in range(num_epochs):
            for i, data in enumerate(self.train_dataloader, 0):
                train_loss = self.train_iter(data)

                if i % self.config["log_interval"] == 0 or\
                        i % self.config["save_interval"] == 0 or\
                        i == len(self.train_dataloader)-1:
                    print(f"Epoch {epoch}, Step {i},\n Training Loss: {train_loss}")
                    valid_errors, _ = self.eval()
                    print(f"Validation Error: {np.average(valid_errors)}")

                if i % self.config["save_interval"] == 0 or i == len(self.train_dataloader)-1:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': epoch,
                                'valid_error': np.average(valid_errors)
                                },
                               self.output_dir + f"ep{epoch}_step{i}.pt"
                               )


class Code2VecTrainer(BaseTrainer):
    """
    Trainer class that initializes a Code2Vec model and
    dataloader to provide embeddings. Generates training programs
    if needed.
    """
    def __init__(self, prog_config, matrix_config, training_config, model_class, progs=None, load_path=None,
            data_cache_path=None, code2vec_dataset=None):
        self.prog_config = prog_config
        self.matrix_config = matrix_config
        self.model_class = model_class
        if progs is None:
            self.progs = list(gen_progs(**self.prog_config))
        else:
            self.progs = progs

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if code2vec_dataset is None:
            self.all_data = initialize_code2vec_dataset(self.progs, **self.matrix_config, load_cache_file=data_cache_path)
        else:
            self.all_data = code2vec_dataset

        train_dataset, test_dataset = torch.utils.data.random_split(self.all_data,
                                                                    training_config["train_split"]
                                                                    )

        self.embedding_size = self.all_data.code2vec_embedding.config.CODE_VECTOR_SIZE
        self.N = self.prog_config["N"]

        # Define model
        model = self.model_class(self.embedding_size, self.N)
        if load_path is not None:
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=training_config["batch_size"],
                                                       shuffle=True
                                                       )
        valid_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=training_config["batch_size"],
                                                       shuffle=False
                                                       )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.001,
                                     betas=(0.9, 0.999))

        super().__init__(model, device, optimizer, train_dataloader, valid_dataloader, training_config)

        with open(self.output_dir + "/train_progs.pkl", "wb") as progs_file:
            pickle.dump(train_dataset.dataset.progs, progs_file)

        with open(self.output_dir + "/valid_progs.pkl", "wb") as progs_file:
            pickle.dump(test_dataset.dataset.progs, progs_file)
