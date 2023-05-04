import numpy as np
import torch
from antlr4 import *

from code2vec.interactive_predict import InteractivePredictor

from antlr.interpreter import MatrixInterpreter

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("import Pkg; Pkg.activate(\"./utils/tensorgames/\")")
from julia import TensorGames


device_cpu = torch.device("cpu")


class Code2VecEmbedding(InteractivePredictor):
    """
    Class that wraps around Code2Vec's model class
    to support programs from a DSL.
    """
    def __init__(self, config, model, prog_tmp_file_path=None):
        super().__init__(config, model)
        if prog_tmp_file_path:
            self.prog_tmp_file_path = prog_tmp_file_path
        else:
            self.prog_tmp_file_path = f"/tmp/prog{np.random.choice(10000000)}.java"

    def embed_program(self, prog):
        """
        Return the embedding (code vector) for a program.
        :param prog: Source code to embed (string).
        :return: Embedding/code vector (numpy array).
        """

        with open(self.prog_tmp_file_path, 'w') as fh:
            fh.write(prog)

        # Run the model
        predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(self.prog_tmp_file_path)
        raw_prediction_results = self.model.predict(predict_lines)
        raw_prediction = raw_prediction_results[0]

        return np.array(raw_prediction.code_vector)


class MatrixProgramEmbeddingDataset(torch.utils.data.Dataset):
    """
    Compute code vectors, ground truth matrices and Nash strategies
    for a set of programs and allow them to be sampled as a dataset.
    """
    def __init__(self, progs, code2vec_args, matrix_args, opts=None, load_cache_file=None):
        self.progs = progs
        self.code2vec_embedding = Code2VecEmbedding(**code2vec_args)

        self.cache = {}
        if load_cache_file is not None:
            self.load_data_from_file(load_cache_file)

        self.interpreter = MatrixInterpreter(**matrix_args)
        self.opts = opts if opts is not None else {}
        super(MatrixProgramEmbeddingDataset, self).__init__()

    def get_matrix(self, prog):
        matrix_np = self.interpreter(InputStream(prog))
        matrix = torch.tensor(matrix_np, dtype=torch.float32, device=device_cpu)
        return matrix_np, matrix

    def process_prog(self, prog):
        matrix_np, matrix = self.get_matrix(prog)
        # Code2vec expects a function
        prog_as_function = f"void f(Matrix A){{\n {prog}\n }}\n"
        code_vector = torch.tensor(
            self.code2vec_embedding.embed_program(prog_as_function),
            dtype=torch.float32,
            device=device_cpu
        )
        # Find a Nash solution for the game
        # Assumed to be zero-sum, so players' matrices are of the form A, -A
        if "nash_solutions" in self.opts:
            res = TensorGames.compute_equilibrium([matrix_np, -matrix_np])
            nash_solutions = torch.tensor(res.x, dtype=torch.float32, device=device_cpu)
            cost = TensorGames.expected_cost(res.x, matrix_np)
            cost_pyt = torch.tensor([cost], dtype=torch.float32, device=device_cpu)

        if "nash_solutions" in self.opts:
            return code_vector, matrix, nash_solutions, cost_pyt
        else:
            return code_vector, matrix

    def __getitem__(self, idx):
        if not (idx in self.cache):
            prog = self.progs[idx]
            self.cache[idx] = self.process_prog(prog)
        return self.cache[idx]

    def save_data(self, save_file_path):
        torch.save([self.progs, self.cache], save_file_path)

    def load_data_from_file(self, load_file_path):
        self.progs, self.cache = torch.load(load_file_path)

    def save_cache(self, save_file_path):
        torch.save(self.cache, save_file_path)

    def load_cache_from_file(self, load_file_path):
        self.cache = torch.load(load_file_path)

    def __len__(self):
        return len(self.progs)


if __name__ == "__main__":
    from data.gen_matrix_progs import gen_matrix_progs_v0 as gen_progs
    N = 16
    num_programs = 5
    num_statements = 2

    progs = gen_progs(N, num_programs, num_statements)

    from code2vec.config import Config
    from code2vec.code2vec import load_model_dynamically

    config = Config(set_defaults=True, load_from_args=True, verify=True)
    c2v_model = load_model_dynamically(config)

    # Create Torch datasets
    dataset_without_nash = MatrixProgramEmbeddingDataset(list(progs),
                                             {"config": config,
                                              "model": c2v_model,
                                              },
                                             {"base_array": np.zeros((N, N))}
                                             )

    for x in dataset_without_nash[0]:
        print(x)

    dataset_with_nash = MatrixProgramEmbeddingDataset(list(progs),
                                                         {"config": config,
                                                          "model": c2v_model,
                                                          },
                                                         {"base_array": np.zeros((N, N))},
                                                         {"nash_solutions": True}
                                                         )

    dataset_with_nash.save_cache('output/test_cache.pt')
    dataset_with_nash_loaded = MatrixProgramEmbeddingDataset(list(progs),
                                  {"config": config,
                                   "model": c2v_model,
                                   },
                                  {"base_array": np.zeros((N, N))},
                                  {"nash_solutions": True},
                                   load_cache_file='output/test_cache.pt',
                                  )
    for x in dataset_with_nash_loaded[0]:
        print(x)
