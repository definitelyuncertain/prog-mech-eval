import numpy as np

import tensorflow as tf

from data.dataloader import MatrixProgramEmbeddingDataset
from data.gen_matrix_progs import gen_matrix_progs_col_stripe as gen_progs

from code2vec.config import Config
from code2vec.code2vec import load_model_dynamically



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

prog_config = dict(N=16,
                   num_programs=1024000,
                   num_statements=3,
                   stripe_size=6,
                   stripe_chance=0.25,
                   )
progs = gen_progs(**prog_config)

config = Config(set_defaults=True, load_from_args=True, verify=True)
c2v_model = load_model_dynamically(config)

base_mat = np.zeros((
        prog_config["N"],
        prog_config["N"],
))

matrix_config = {
    "matrix_args": {"base_array": base_mat},
    "opts": {"nash_solutions": True},
}

dataset_with_nash = MatrixProgramEmbeddingDataset(list(progs),
                                                     {"config": config,
                                                      "model": c2v_model,
                                                      },
                                                     {"base_array": np.zeros(
                                                         (prog_config["N"],
                                                          prog_config["N"]
                                                          ))},
                                                     {"nash_solutions": True},
                                                  load_cache_file="models/progs_data_rand_col_stripe_256k.pt",
                                                  )

for i, _ in enumerate(dataset_with_nash):
    if i % 10000 == 0 or i == len(dataset_with_nash)-1:
        print(f"Processed {i}")
        dataset_with_nash.save_data('models/progs_data_rand_col_stripe_256k.pt')
