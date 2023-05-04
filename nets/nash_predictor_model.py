import numpy as np
import torch

class PlayerNashPredictor(torch.nn.Module):
    """
    Per-agent policy model using an MLP architecture to
    regress from program embeddings to policy probabilities.
    Outputs log-probabilities using a log_softmax activation.
    """
    def __init__(self, embedding_size, N):
        super().__init__()
        self.embedding_size = embedding_size
        self.N = N
        self.layer_sizes = [self.embedding_size, 128, 128, self.N]
        modules = []
        for i in range(len(self.layer_sizes)-2):
            fc = torch.nn.Linear(self.layer_sizes[i],
                                 self.layer_sizes[i+1]
                                 )
            modules.append(fc)
            if i > 0:
                modules.append(torch.nn.LeakyReLU())

        modules.append(torch.nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x):
        logits = self.mlp(x)
        # Get the Nash strategies as log-probs of the logits for each agent
        logpi = torch.nn.functional.log_softmax(logits, dim=1)
        return logpi


class PlayerNashPredictorConv(torch.nn.Module):
    """
    Per-agent policy model using a convolutional architecture to
    regress from program embeddings to policy probabilities.
    First upsamples the embedding into image channels, and then
    downsamples it using regular conv layers.
    Outputs log-probabilities using a log_softmax activation.
    """
    def __init__(self, embedding_size, N):
        super().__init__()
        self.embedding_size = embedding_size
        self.N = N

        self.lin = torch.nn.Linear(self.embedding_size,
                                   4 * self.embedding_size)
        channels = [self.embedding_size, 128, 64, 32]
        modules = []
        for i in range(len(channels) - 1):
            tconv = torch.nn.ConvTranspose2d(channels[i],
                                             channels[i + 1],
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             output_padding=1
                                             )
            layers = [tconv,
                      torch.nn.LeakyReLU(),
                      ]
            layers.append(torch.nn.BatchNorm2d(channels[i + 1]))

            modules.append(
                torch.nn.Sequential(*layers)
            )
        self.decoder = torch.nn.Sequential(*modules)

        channels = [32, 64, 128, 256]
        modules = []
        for i in range(len(channels) - 1):
            conv = torch.nn.Conv2d(channels[i],
                                      out_channels=channels[i+1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1
                                    )
            layers = [conv,
                      torch.nn.LeakyReLU(),
                      torch.nn.BatchNorm2d(channels[i + 1])
                      ]
            modules.append(
                torch.nn.Sequential(*layers)
            )
        self.predictor = torch.nn.Sequential(*modules)
        self.logit_head = torch.nn.Linear(4 * channels[-1],
                                          self.N)

    def forward(self, x):
        l = self.lin(x).view(-1, self.embedding_size, 2, 2)
        y = self.predictor(self.decoder(l))
        logits = self.logit_head(torch.flatten(y, start_dim=1))
        # Get the Nash strategies as log-probs of the logits for each agent
        logpi = torch.nn.functional.log_softmax(logits, dim=1)
        return logpi


class NashPredictor(torch.nn.Module):
    """
    Wraps two MLP policies for 2 players to return
    both players' strategies (log-probabilities) for a
    given program embedding.
    """
    def __init__(self, embedding_size, N):
        super().__init__()
        self.pi1, self.pi2 = (PlayerNashPredictor(embedding_size, N),
                              PlayerNashPredictor(embedding_size, N)
                              )

    def forward(self, x):
        pi1, pi2 = self.pi1(x), self.pi2(x)
        return pi1, pi2


class NashPredictorConv(torch.nn.Module):
    """
        Wraps two convolutional policies for 2 players to return
        both players' strategies (log-probabilities) for a
        given program embedding.
    """
    def __init__(self, embedding_size, N):
        super().__init__()
        self.pi1, self.pi2 = (PlayerNashPredictorConv(embedding_size, N),
                              PlayerNashPredictorConv(embedding_size, N)
                              )

    def forward(self, x):
        pi1 = self.pi1(x)
        pi2 = self.pi2(x)
        return pi1, pi2


if __name__ == "__main__":
    from data.gen_matrix_progs import gen_matrix_progs_v0 as gen_progs

    N = 16
    num_programs = 1
    num_statements = 2

    progs = gen_progs(N, num_programs, num_statements)

    from utils.utils import initialize_code2vec_dataset

    dataset = initialize_code2vec_dataset(
        progs,
        matrix_args={"base_array": np.zeros((N, N))},
        opts={"nash_solutions": True}
    )

    model = NashPredictor(
        dataset.code2vec_embedding.config.CODE_VECTOR_SIZE,
        N
    )

    code_vector, matrix, nash_solutions = dataset[0][:3]

    print(model(code_vector[None, :]))

    model = NashPredictorConv(
        dataset.code2vec_embedding.config.CODE_VECTOR_SIZE,
        N
    )

    print(model(code_vector[None, :]))
