import torch


class MatrixPredictor(torch.nn.Module):
    """
    Model that predicts the output matrix based on the
    input embedding (Code Vector). Uses a series of
    transposed conv layers to upsample the embedding into
    channels of images, and convolutions to get the
    final matrix reconstruction.
    """
    def __init__(self, embedding_size, N):
        super().__init__()
        self.embedding_size = embedding_size
        self.N = N
        self.lin = torch.nn.Linear(self.embedding_size,
                                   4 * self.embedding_size)
        channels = [self.embedding_size, 128, 64, 32]
        modules = []
        for i in range(len(channels)-1):
            tconv = torch.nn.ConvTranspose2d(channels[i],
                                             channels[i+1],
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             output_padding=1
                                             )
            layers = [tconv,
                      torch.nn.LeakyReLU(),
                      ]
            layers.append(torch.nn.BatchNorm2d(channels[i+1]))


            modules.append(
                torch.nn.Sequential(*layers)
            )

        layers = []
        layers.append(torch.nn.Conv2d(channels[-1],
                                      out_channels=channels[-1] // 2,
                                      kernel_size=3,
                                      padding=1
                                      ),
                      )
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm2d(channels[-1] // 2))
        layers.append(torch.nn.Conv2d(channels[-1]//2,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding=1
                                      ),
                      )
        layers.append(torch.nn.ReLU())
        modules.append(
            torch.nn.Sequential(*layers)
        )
        self.decoder = torch.nn.Sequential(*modules)

    def forward(self, x):
        l = self.lin(x)
        m0 = l.view(-1, self.embedding_size, 2, 2)
        output = self.decoder(m0)
        return output.view(-1, output.shape[2], output.shape[3])

