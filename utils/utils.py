import sys

from data.dataloader import MatrixProgramEmbeddingDataset

from code2vec.config import Config
from code2vec.code2vec import load_model_dynamically


def initialize_code2vec_dataset(progs, model_load_path=None, **kwargs):
    """
    Helper function to configure and load a Code2Vec model, and create a
    dataset object using it.

    :param progs: List of programs for the dataset.
    :param model_load_path: Path to the Code2Vec model.
    :return: Matrix dataset (MatrixProgramEmbeddingDataset) object.
    """
    if model_load_path is None:
        model_load_path = '../data/java14m_model/models/java14_model/saved_model_iter8.release'
    # Have to pretend that the arguments were in command-line
    # before calling code2vec loader
    sys.argv.extend(['--load', model_load_path, '--export_code_vectors'])

    config = Config(set_defaults=True, load_from_args=True, verify=True)
    c2v_model = load_model_dynamically(config)

    # Create Torch datasets
    dataset = MatrixProgramEmbeddingDataset(list(progs),
                                             {"config": config,
                                              "model": c2v_model,
                                              },
                                             **kwargs
                                             )

    return dataset
