import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import csv

from DataLoader import Dataloader
from Casper_model import Casper


class Casper_trainer:

    def __init__(self, dataloader):
        super(Casper_trainer, self).__init__()

        # load data using Dataloader class

        self.X, self.Y, self.vali_input, self.vali_target, self.test_input, self.test_target = dataloader.train_smaller_input, dataloader.train_smaller_target, dataloader.vali_input, dataloader.vali_target, dataloader.test_input, dataloader.test_target

        # set hyper parameters
        self.n_input = 6
        self.n_output = 2

        # 3 different learning rates for different parts of the network according to the definition of Casper
        self.l1 = 0.2
        self.l2 = self.l1 / 4
        self.l3 = self.l2 / 50
        # p is used to determine the interval between checkpoints while training, larger p results in less checkpoints.
        self.p = 5
        # self.p = 3

        # k is the maximum number of hidden neurons, once Casper's hidden units is about to reach to k+1, training stops.
        # self.k = 5
        # self.k = 10
        self.k = 15
        # self.k = 25
        # self.k = 50
        # self.k = 100

        self.n_epochs = 5000
        # self.n_epochs = 15000
        # self.n_epochs = 30000
        self.batch_size = 80
        self.batch_idx = list(range(self.batch_size))

    def train_validate(self):
        # load the pre-trained DNN model from file for initialization
        with open('DNN_model.pkl', 'rb') as f:
            DNN_model = pickle.load(f)

        # initialize the network and some basic parameter-settings
        net = Casper(DNN_model, self.n_input, self.n_output, self.l1, self.l2, self.l3)

        optimizer = torch.optim.RMSprop(net.parameters(), lr=self.l1, momentum=0.1)
        checkpoint = 15 + self.p * net.hidden_counts

        previous_loss = float('inf')  # used to decided when to add a new neuron
        stop = False  # a simple indicator used to decided when to stop training
        final_epoch = 0

        # data records used to plot figures
        plt_loss_train = []  # train loss
        plt_loss_vali1 = []  # another version of train loss used for comparison with validation loss
        plt_loss_vali2 = []  # validation loss
        plt_acc_vali = []  # validation accuracy

        plt_loss_train.append(previous_loss)
        plt_loss_vali1.append(previous_loss)
        plt_loss_vali2.append(previous_loss)

        # this part is to add the first validation accuracy to plt_acc_vali
        # when there is no new neurons added to the network, that is,
        # the validation accuracy of the pre-trained DNN, which will be used
        # as the initialization of the Casper network.
        total_v = 0
        correct_v = 0
        for i in range(int(len(self.vali_input) / self.batch_size)):

            # get some data from the validation set
            idx = []
            for j in self.batch_idx:
                idx.append(i + j)
            x_vali = self.vali_input[idx]
            y_vali = self.vali_target[idx]

            # preform the validation and record the accuracy
            y_pred_vali = DNN_model(x_vali)

            _, predicted_v = torch.max(y_pred_vali, 1)
            total_v = total_v + y_pred_vali.size(0)
            correct_v = correct_v + sum(predicted_v.data.numpy() == y_vali.data.numpy())

        original_DNN_acc = (correct_v / total_v) * 100
        plt_acc_vali.append(original_DNN_acc)

        # This list is used to store the status of the previous 100
        # networks of the current network. Because when Casper stops,
        # the current network state is often not the best
        # under the current k value, so select the 15th state in this list, that is,
        # the 75th state in front of the current network as the output
        previous_nets = []

        # These two variables are used to control the maximum number of epochs
        # that Casper stays in the same network state. The setting of the following code is:
        # if the training stays on the same hidden neuron for more than 30 checkpoints, then add a neuron.
        previous_n_hidden = 0
        times_n_hidden_not_change = 1

        print(
            '\nTraining for the Casper network starts,\n'
            'parameters have been set as: maximum hidden neurons: %d, maximum epochs: %d, p: %d, k: %d\n' % (
                self.k, self.n_epochs, self.p, self.k))

        for epoch in range(self.n_epochs):

            # initialize records
            total = 0
            correct = 0
            total_loss = 0

            for i in range(int(len(self.X) / self.batch_size)):

                # select some data form training set according to batch_size
                idx = []
                for j in self.batch_idx:
                    idx.append(i + j)
                x = self.X[idx]
                y = self.Y[idx]

                # some regular processes
                optimizer.zero_grad()
                output = net(x)
                loss = nn.CrossEntropyLoss()(output, y)  # use CrossEntropyLoss, no need to apply softmax
                loss.backward()
                optimizer.step()

                # store the current net in the list which contains the previous 100 nets
                if len(previous_nets) == 100:
                    previous_nets.remove(previous_nets[0])
                previous_nets.append(net)

                # if this epoch is a checkpoint, record the loss and accuracy
                if epoch == checkpoint:
                    _, predicted = torch.max(output, 1)
                    total = total + predicted.size(0)
                    correct = correct + sum(predicted.data.numpy() == y.data.numpy())
                    total_loss = total_loss + loss

            # also if this epoch is a checkpoint, check if new neuron should be added
            if epoch == checkpoint:

                # update the checkpoint
                N = net.hidden_counts
                checkpoint += 15 + self.p * N

                # record the previous number of hidden units in the network
                if N == previous_n_hidden:
                    times_n_hidden_not_change += 1
                previous_n_hidden = N

                # once Casper's hidden units is about to reach to k+1, training stops.
                if net.hidden_counts == self.k:
                    stop = True
                    final_epoch = epoch

                # add a neuron when the loss is not decreasing enough (0.0015)
                if (previous_loss <= total_loss + 0.0015) & (previous_loss > total_loss):

                    # is training stops, select the 15th network from the previous network status
                    if stop:
                        net = previous_nets[14]
                        break

                    # the training continue, neuron is about to be added, but we should first validating the loss.
                    else:
                        all_vali_loss = 0
                        total_v = 0
                        correct_v = 0
                        for i in range(int(len(self.vali_input) / self.batch_size)):

                            # get some data from the validation set
                            idx = []
                            for j in self.batch_idx:
                                idx.append(i + j)
                            x_vali = self.vali_input[idx]
                            y_vali = self.vali_target[idx]

                            # preform the validation and record the loss and accuracy
                            y_pred_vali = previous_nets[14](x_vali)
                            vali_loss = nn.CrossEntropyLoss()(y_pred_vali, y_vali)
                            all_vali_loss = all_vali_loss + vali_loss

                            _, predicted_v = torch.max(y_pred_vali, 1)
                            total_v = total_v + y_pred_vali.size(0)
                            correct_v = correct_v + sum(predicted_v.data.numpy() == y_vali.data.numpy())

                        # save the accuracy and loss for plotting
                        vali_acc = (correct_v / total_v) * 100
                        plt_acc_vali.append(vali_acc)
                        plt_loss_vali1.append(all_vali_loss.item())
                        plt_loss_vali2.append(total_loss.item())

                        # reset the parameter
                        times_n_hidden_not_change = 1

                        # after validation, add a new neuron and get the new optimizer
                        net.add_neuron()
                        optimizer = net.update_lr_opt()

                # this part is used to add a unit when the network has been stuck in the same state for 30 times
                elif times_n_hidden_not_change == 30:
                    if stop:
                        net = previous_nets[14]
                        break

                    print('\nThe network has been stuck in the same state for 30 times, thus added a new neuron.\n')

                    # this is the same validation process with the above
                    all_vali_loss = 0
                    total_v = 0
                    correct_v = 0
                    for i in range(int(len(self.vali_input) / self.batch_size)):

                        # get some data from the validation set
                        idx = []
                        for j in self.batch_idx:
                            idx.append(i + j)
                        x_vali = self.vali_input[idx]
                        y_vali = self.vali_target[idx]

                        # preform the validation and record the loss
                        y_pred_vali = previous_nets[14](x_vali)
                        vali_loss = nn.CrossEntropyLoss()(y_pred_vali, y_vali)
                        all_vali_loss = all_vali_loss + vali_loss

                        _, predicted_v = torch.max(y_pred_vali, 1)
                        total_v = total_v + y_pred_vali.size(0)
                        correct_v = correct_v + sum(predicted_v.data.numpy() == y_vali.data.numpy())

                    # save the accuracy and loss for plotting
                    vali_acc = (correct_v / total_v) * 100
                    plt_acc_vali.append(vali_acc)
                    plt_loss_vali1.append(all_vali_loss.item())
                    plt_loss_vali2.append(total_loss.item())

                    # reset the parameter
                    times_n_hidden_not_change = 1

                    net.add_neuron()
                    optimizer = net.update_lr_opt()

                # reset the list
                previous_nets = []

                # save the loss for plotting
                plt_loss_train.append(total_loss.item())

                # update the previous loss
                previous_loss = total_loss

                print('The number of hidden neurons in the Casper network is now: ', net.hidden_counts)
                print('Casper: Epoch [%d/%d], Training loss: %.4f, Training accuracy: %.2f %%'
                      % (epoch + 1, self.n_epochs,
                         total_loss, 100 * correct / total))

        # plot 1st figure: loss changes with the increase of checkpoints
        plt.figure()
        plt.plot(plt_loss_train)
        plt.xlabel('Checkpoints')
        plt.ylabel('Loss')
        plt.title('Training Loss for Casper')
        plt.show()
        plt.close()
        print('\nA figure of training loss has been generated')

        # plot 2nd figure: the validation loss and training loss change with the increase of hidden neurons added
        # this picture is used to determine the setting of the hyper parameter k
        axis = []
        for i in range(len(plt_loss_vali1)):
            axis.append(list(range(len(plt_loss_vali1)))[i] + 1)

        plt.plot(axis, plt_loss_vali1, linestyle="--", label="vali loss")
        plt.plot(axis, plt_loss_vali2, label="train loss")
        plt.xlabel('Added neurons')
        plt.ylabel('Loss')
        plt.title('Validation Loss for Casper')
        plt.legend()
        plt.show()
        plt.close()
        print('A figure of validation loss has been generated')

        # plot 3rd figure: Validation accuracy for Casper,
        # this picture is used to determine the setting of the hyper parameter k
        plt.plot(plt_acc_vali)
        plt.xlabel('Added neurons')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy for Casper')
        plt.show()
        plt.close()
        print('A figure of validation accuracy has been generated')

        # save the model to file
        with open('Casper_model.pkl', 'wb') as f:
            pickle.dump(net, f)

        print(
            '\nThe training for the Casper network is finished at epoch %d, model has been saved to Casper_model.pkl.' % final_epoch)

    def test_comparison(self):

        # load the model from file
        with open('Casper_model.pkl', 'rb') as f:
            net = pickle.load(f)

        print('\nTesting for the Casper starts')

        all_test_pred = []
        all_test_loss = 0
        correct_test = 0
        for i in range(int(len(self.test_input) / self.batch_size)):

            # get some data from testing set
            idx = []
            for j in self.batch_idx:
                idx.append(i + j)
            x_test = self.test_input[idx]
            y_test = self.test_target[idx]

            # regular processes
            y_pred_test = net(x_test)
            test_loss = nn.CrossEntropyLoss()(y_pred_test, y_test)
            all_test_loss = all_test_loss + test_loss

            _, predicted_test = torch.max(y_pred_test, 1)
            for j in range(len(predicted_test.data)):
                all_test_pred.append(predicted_test.data[j])

            correct_test = correct_test + sum(predicted_test.data.numpy() == y_test.data.numpy())

        final_acc_Casper = 100 * correct_test / len(self.test_input)

        print("\nThe confusion matrix of testing result of the Casper network is:")
        print(
            Dataloader.plot_confusion(len(self.test_target.data), self.n_output, all_test_pred, self.test_target.data))
        print('\nCasper: Testing loss: %.4f, Testing accuracy: %.2f %%' % (
            all_test_loss, final_acc_Casper))
        print('Testing for the Casper network is finished')

        # for comparison
        # read in original testing accuracy of DNN from file
        f = open('DNN_test_accuracy.txt', 'r')
        DNN_test_acc = float(f.read())
        f.close()

        # read in original testing accuracy of DNN with full training set from file
        f = open('DNN_with_full_training_set_test_accuracy.txt', 'r')
        DNN_test_acc_with_full_training_set = float(f.read())
        f.close()

        print(
            "\nComparison: Using different training data from the same dataset,\n"
            "the testing accuracy of the original DNN used as initialization of Casper is: %.2f %%,\n"
            "the testing accuracy of the original DNN trained with full dataset is: %.2f %%,\n"
            "the testing accuracy of the modified DNN is: %.2f %%" % (
                DNN_test_acc, DNN_test_acc_with_full_training_set, final_acc_Casper))

        print(
            "\nThe testing accuracy of the second training stage has decreased %.2f %% than the first stage." % (
                    DNN_test_acc - final_acc_Casper))

        print(
            "The testing accuracy of the DNN structure has approximately dropped %.2f %% after being modified by\n"
            "the Casper algorithm with another %d hidden neurons added" % (
                DNN_test_acc_with_full_training_set - final_acc_Casper, net.hidden_counts))

        # save results for future use
        with open("plot_comparison_result.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([DNN_test_acc, DNN_test_acc_with_full_training_set, final_acc_Casper])
        csvfile.close()
