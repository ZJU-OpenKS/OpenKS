from torch import nn
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.hidden_dim = opt["hidden_dim"]
        self.num_class = opt["num_classes"]
        self.linear = nn.Linear(self.hidden_dim, self.num_class)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight)  # initialize linear layer

    def forward(self, inputs):
        logits = self.linear(inputs)
        return logits


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims, output_dims),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """Forward the discriminator."""
        out = self.layer(inputs)
        return out
