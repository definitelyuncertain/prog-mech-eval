import torch

from nets.nash_predictor_model import NashPredictorConv
from trainers.trainers import Code2VecTrainer


class Code2VecNashBCTrainer(Code2VecTrainer):
    """
    Trainer for the Nash strategy learning experiment.
    Provides the code vector as input to the model and
    computes the loss as KL divergence between the policy
    probabilities and the ground truth nash strategies.
    """
    def __init__(self, prog_config, matrix_config, training_config, model_class=NashPredictorConv, **kwargs):
        super().__init__(prog_config, matrix_config, training_config, model_class=model_class, **kwargs)

    def prepare_data(self, data):
        code_vectors, matrices, nash, _ = data
        inputs = (code_vectors,)
        ground_truth = (nash[:, 0, :], nash[:, 1, :])

        return inputs, ground_truth

    def loss_function(self, model_outputs, ground_truth):
        nash1, nash2 = ground_truth # Targets for KL loss are direct probabilities
        nash1_pred, nash2_pred = model_outputs

        return torch.nn.functional.kl_div(nash1_pred, nash1, reduction="batchmean") + \
               torch.nn.functional.kl_div(nash2_pred, nash2, reduction="batchmean")