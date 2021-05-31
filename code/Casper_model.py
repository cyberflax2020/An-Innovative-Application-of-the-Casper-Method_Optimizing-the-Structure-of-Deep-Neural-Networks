import torch
from torch import nn


class Casper(nn.Module):

    # initialize the net work using a trained DNN
    def __init__(self, DNN_model, n_input, n_output, l1, l2, l3):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        # a classical Casper has 3 learning rates
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.hidden_counts = 0
        self.hiddens = nn.ModuleList()
        self.outputs = nn.ModuleList()

        # instead of use a empty fully connected network, use a trained DNN as initialization
        self.function = DNN_model

    # a function that adds one neuron to the existing network
    def add_neuron(self):
        self.hiddens.append(nn.Sequential(
            # nn.Dropout(p=0.005),
            nn.Linear(self.n_input + self.hidden_counts, 1),
            # linked with the previous added neurons and the input neurons
            # nn.Sigmoid(),
            # nn.ReLU(),
            nn.LeakyReLU()  # activation function
        ))
        self.outputs.append(nn.Sequential(
            nn.Linear(1, self.n_output)  # linked with the out put neurons
        ))
        self.hidden_counts += 1

    # once a new neuron is added, the optimizer with a learning rate setting should be updated
    # this function returns an updated optimizer
    def update_lr_opt(self):
        # if this is the first neuron to be added, return the initialized the optimizer with l1
        if self.hidden_counts - 1 == 0:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.l1, momentum=0.1)
        else:
            # this part is to decide which parts should apply what learning rates according to the Casper definition
            all_paras = list(self.parameters())
            l1_paras = list(self.hiddens[-1].parameters())
            l2_paras = list(self.outputs[-1].parameters())
            l3_paras = []
            idx = []
            for para in l1_paras:
                idx.append(id(para))
            for para in l2_paras:
                idx.append(id(para))
            for para in all_paras:
                if id(para) not in idx:
                    l3_paras.append(para)

            # update the optimizer using the new learning rate setting
            optimizer = torch.optim.RMSprop(
                [{'params': l1_paras, 'lr': self.l1}, {'params': l2_paras, 'lr': self.l2},
                 {'params': l3_paras, 'lr': self.l3}], lr=self.l1,
                momentum=0.1)

        return optimizer

    def forward(self, x):
        # first calculate from the input layer to the hidden layer
        hidden_out = []
        for i in range(self.hidden_counts):
            if len(hidden_out) != 0:
                hidden_out.append(self.hiddens[i](torch.cat([x] + hidden_out[:i], dim=1)))

            else:
                hidden_out.append(self.hiddens[i](x))

        # then calculate from the hidden layer to the output layer
        out = [self.function(x)]

        # print(x.shape)
        # print(self.function(x).shape)

        for i in range(self.hidden_counts):
            out.append(self.outputs[i](hidden_out[i]))
        sum_output = sum(out)

        # note that we use CrossEntropyLoss as loss to do classification, so no need to apply softmax.
        return sum_output
