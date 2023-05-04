import numpy as np

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to .yaml configuration file")
args = parser.parse_args()

config_file_path = args.config
print(f'Using config {config_file_path}')

import tensorflow as tf

from trainers.matrix_trainer import Code2VecMatrixTrainer
from trainers.nash_trainer import Code2VecNashBCTrainer

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


with open(config_file_path, 'r') as stream:
    cfg = yaml.safe_load(stream)
    stream.close()


prog_config = dict(N=16,                  # Size of the game matrices
                   num_programs=1024000,  # Number of programs in the dataset
                   num_statements=3,      # Number of statements in each program
                   stripe_size=6,         # Size of vertical stripes of ones
                   )

matrix_config = {
    "matrix_args": {
        "base_array": np.zeros((          # Base matrix game that programs modify
            cfg['prog_config']["N"],
            cfg['prog_config']["N"],
    ))},
    "opts": {"nash_solutions": True},     # Whether to compute Nash solutions
}

training_config = {
        "output_dir": "models/test/",
        "log_interval": 500,
        "save_interval": 500,
        "batch_size": 256,
        "train_split": [0.95, 0.05],
    }

trainer_args = dict(
    # Optional path for loading cached dataset
    # data_cache_path="models/progs_data_rand_col_stripe_1M.pt"
)


# Update above configs with yaml config
for key in ['prog_config', 'matrix_config', 'training_config', 'trainer_args']:
    if key in cfg:
        eval(key).update(cfg[key])

experiment_type = (cfg['experiment_type'])
num_epochs = cfg['num_epochs']


trainers = {
    'nash': Code2VecNashBCTrainer,   # Nash prediction
    'matrix': Code2VecMatrixTrainer  # Matrix prediction
}

if experiment_type not in trainers:
    raise ValueError

TrainerClass = trainers[experiment_type]


# Initialize trainer
trainer = TrainerClass(
    prog_config,
    matrix_config,
    training_config,
    **trainer_args
)

# Run training loop
trainer.train_loop(num_epochs=num_epochs)
