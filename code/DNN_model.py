from torch import nn


# This is a three-layer DNN model.
# Except the output layer, use the Leaky Relu function as the activation.
class DNN(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(DNN, self).__init__()

        self.function = nn.Sequential(
            # nn.Dropout(p=0.001),
            nn.Linear(n_input, n_hidden1),  # The first hidden layer
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Sigmoid(),

            # nn.Dropout(p=0.001),
            nn.Linear(n_hidden1, n_hidden2),  # The second hidden layer
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Sigmoid(),

            nn.Linear(n_hidden2, n_hidden3),  # The third hidden layer
            nn.LeakyReLU(),
            # nn.ReLU(),

            # nn.Linear(n_hidden3, n_hidden4),
            # nn.LeakyReLU(),

            nn.Linear(n_hidden3, n_output)  # Don't use activation because using CrossEntropy
        )

    def forward(self, x):
        # out = nn.Dropout(p=0.001)(x)
        out = self.function(x)
        return out
