import torch

from nets.matrix_predictor_model import MatrixPredictor
from trainers.trainers import Code2VecTrainer


class Code2VecMatrixTrainer(Code2VecTrainer):
    """
    Trainer for the matrix prediction experiment.
    Provides the code vector as input to the model and
    computes the MSE loss between the ground truth
    matrices and the model's reconstruction.
    """
    def __init__(self, prog_config, matrix_config, training_config, **kwargs):
        super().__init__(prog_config, matrix_config, training_config, model_class=MatrixPredictor, **kwargs)

    def prepare_data(self, data):
        code_vectors, matrices, _, _ = data
        inputs = (code_vectors,)
        ground_truth = (matrices,)

        return inputs, ground_truth

    def loss_function(self, model_outputs, ground_truth):
        matrices_gt = ground_truth[0]
        matrices_pred = model_outputs

        return torch.nn.functional.mse_loss(matrices_pred, matrices_gt)