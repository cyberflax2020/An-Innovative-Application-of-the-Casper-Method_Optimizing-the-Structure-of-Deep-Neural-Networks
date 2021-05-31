import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle

from DataLoader import Dataloader
from DNN_model import DNN


class DNN_trainer:

    def __init__(self, dataloader):
        super(DNN_trainer, self).__init__()

        # load data using Dataloader class
        self.X, self.Y, self.vali_input, self.vali_target, self.test_input, self.test_target = dataloader.train_larger_input, dataloader.train_larger_target, dataloader.vali_input, dataloader.vali_target, dataloader.test_input, dataloader.test_target

        # set hyper parameters
        self.n_input = 6
        self.n_output = 2
        self.n_hidden1 = 120
        self.n_hidden2 = 120
        self.n_hidden3 = 4
        # self.n_hidden4 = 4

        self.lr = 0.2
        self.n_epochs = 2000

    # train and save the DNN model to file
    def train(self):
        net = DNN(self.n_input, self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_output)

        loss_func = torch.nn.CrossEntropyLoss()  # use CrossEntropyLoss for classification

        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)

        all_losses = []

        print('\nTraining for the DNN starts, total epochs: %d\n' % self.n_epochs)

        for epoch in range(self.n_epochs):
            Y_pred = net(self.X)
            loss = loss_func(Y_pred, self.Y)
            all_losses.append(loss.item())

            if epoch % 50 == 0:
                _, predicted = torch.max(Y_pred, 1)

                # calculate and print accuracy
                total = predicted.size(0)
                correct = predicted.data.numpy() == self.Y.data.numpy()

                print('DNN: Epoch [%d/%d] Training loss: %.4f  Training accuracy: %.2f %%'
                      % (epoch + 1, self.n_epochs, loss.item(), 100 * sum(correct) / total))

            net.zero_grad()
            loss.backward()
            optimiser.step()

        # plot the training loss
        plt.figure()
        plt.plot(all_losses)
        plt.xlabel('Checkpoints')
        plt.ylabel('Loss')
        plt.title('Training Loss for DNN')
        plt.show()
        plt.close()

        # save the model to file
        with open('DNN_model.pkl', 'wb') as f:
            pickle.dump(net, f)

        print('\nA figure of training loss has been generated')
        print('\nThe training for the DNN is finished, model has been saved to DNN_model.pkl.')

    def test(self):
        # load the model from file
        with open('DNN_model.pkl', 'rb') as f:
            net = pickle.load(f)

        print('\nTesting for the DNN starts')

        outputs = net(self.test_input)
        _, predicted = torch.max(outputs, 1)
        test_loss = nn.CrossEntropyLoss()(outputs, self.test_target)

        total = predicted.size(0)
        correct = predicted.data.numpy() == self.test_target.data.numpy()
        final_acc_DNN = 100 * sum(correct) / total

        print('\nThe confusion matrix of testing result of the DNN is')
        print(Dataloader.plot_confusion(self.test_input.shape[0], self.n_output, predicted.long().data,
                                        self.test_target.data))
        print('\nDNN: Testing loss: %.4f, Testing accuracy: %.2f %%' % (test_loss, final_acc_DNN))
        print('Testing for DNN is finished')

        # save testing accuracy of DNN to file for other class to use
        f = open('DNN_test_accuracy.txt', 'w')
        f.write(str(final_acc_DNN))
        f.close()


# This part below is basically the same with the above but using the full_training_set for comparison
# The evaluation of this part has been simplified.
class DNN_trainer_with_full_training_set:

    def __init__(self, dataloader):
        super(DNN_trainer_with_full_training_set, self).__init__()

        # load data using Dataloader class
        self.X, self.Y, self.vali_input, self.vali_target, self.test_input, self.test_target = dataloader.train_input, dataloader.train_target, dataloader.vali_input, dataloader.vali_target, dataloader.test_input, dataloader.test_target

        # set hyper parameters
        self.n_input = 6
        self.n_output = 2
        self.n_hidden1 = 120
        self.n_hidden2 = 120
        self.n_hidden3 = 4
        # self.n_hidden4 = 4

        self.lr = 0.2
        self.n_epochs = 2000

    # train and save the DNN model to file
    def train(self):
        net = DNN(self.n_input, self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_output)

        loss_func = torch.nn.CrossEntropyLoss()  # use CrossEntropyLoss for classification

        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)

        all_losses = []

        print('\nTraining for the DNN with full training set starts, total epochs: %d\n' % self.n_epochs)

        for epoch in range(self.n_epochs):
            Y_pred = net(self.X)
            loss = loss_func(Y_pred, self.Y)
            all_losses.append(loss.item())

            if epoch % 50 == 0:
                _, predicted = torch.max(Y_pred, 1)

                # calculate and print accuracy
                total = predicted.size(0)
                correct = predicted.data.numpy() == self.Y.data.numpy()

                print('DNN with full training set: Epoch [%d/%d] Training loss: %.4f  Training accuracy: %.2f %%'
                      % (epoch + 1, self.n_epochs, loss.item(), 100 * sum(correct) / total))

            net.zero_grad()
            loss.backward()
            optimiser.step()

        # save the model to file
        with open('DNN_model_with_full_training_set.pkl', 'wb') as f:
            pickle.dump(net, f)

        print(
            '\nThe training for the DNN with full training set is finished, model has been saved to DNN_model_with_full_training_set.pkl.')

    def test(self):
        # load the model from file
        with open('DNN_model_with_full_training_set.pkl', 'rb') as f:
            net = pickle.load(f)

        print('\nTesting for the DNN with full training set starts')

        outputs = net(self.test_input)
        _, predicted = torch.max(outputs, 1)
        test_loss = nn.CrossEntropyLoss()(outputs, self.test_target)

        total = predicted.size(0)
        correct = predicted.data.numpy() == self.test_target.data.numpy()
        final_acc_DNN = 100 * sum(correct) / total

        print('\nThe confusion matrix of testing result of the DNN with full training set is')
        print(Dataloader.plot_confusion(self.test_input.shape[0], self.n_output, predicted.long().data,
                                        self.test_target.data))
        print(
            '\nDNN with full training set: Testing loss: %.4f, Testing accuracy: %.2f %%' % (test_loss, final_acc_DNN))
        print('Testing for DNN with full training set is finished')

        # save testing accuracy of DNN to file for other class to use
        f = open('DNN_with_full_training_set_test_accuracy.txt', 'w')
        f.write(str(final_acc_DNN))
        f.close()
