import os
import sys
import csv
import matplotlib.pyplot as plt

from DataLoader import Dataloader
from DNN_train_evaluation import DNN_trainer
from DNN_train_evaluation import DNN_trainer_with_full_training_set
from Casper_train_evaluation import Casper_trainer
from Logger import Logger

'''
The following code is to run the code in main.py multiple times to 
obtain the data and generate the figure used in the article. 
This part of the code is very time-consuming, please run it with caution.
'''

try:
    os.remove('plot_comparison_result.csv')
    os.remove('console_record.txt')
except BaseException as e:
    print(e)
    print("\nCreated new files")

# this part is a simple report generator which saves all console info in to a .txt file
sys.stdout = Logger("console_record.txt")
print("\nRunning get_average_result_comparison.py")

iteration = 10

# load and preprocess the data
dataloader = Dataloader(r'Anger.xlsx')
dataloader.load_and_preprocess()

testing_acc_DNN = []
testing_acc_DNN_with_full_training_set = []
testing_acc_Casper = []
result = []

for i in range(iteration):
    print("\nThis is iteration %d" % (i + 1))
    dnn_trainer = DNN_trainer(dataloader)
    dnn_trainer.train()
    dnn_trainer.test()

    dnn_trainer_with_full_training_set = DNN_trainer_with_full_training_set(dataloader)
    dnn_trainer_with_full_training_set.train()
    dnn_trainer_with_full_training_set.test()

    casper_trainer = Casper_trainer(dataloader)
    casper_trainer.train_validate()
    casper_trainer.test_comparison()

with open("plot_comparison_result.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        testing_acc_DNN.append(float(line[0]))
        testing_acc_DNN_with_full_training_set.append(float(line[1]))
        testing_acc_Casper.append(float(line[2]))
        result.append(float(line[1]) - float(line[2]))
csvfile.close()

axis = range(iteration)

plt.figure()

plt.plot(axis, testing_acc_DNN, linestyle="--",
         label="Average testing accuracy of DNN used as initialization of Casper: %.2f %%" % (sum(testing_acc_DNN) / len(testing_acc_DNN)))
plt.plot(axis, testing_acc_DNN_with_full_training_set, linestyle="-.",
         label="Average testing accuracy of DNN trained with full dataset: %.2f %%" % (
                 sum(testing_acc_DNN_with_full_training_set) / len(testing_acc_DNN_with_full_training_set)))
plt.plot(axis, testing_acc_Casper,
         label="Average testing accuracy of modified DNN with Casper algorithm: %.2f %%" % (
                 sum(testing_acc_Casper) / len(testing_acc_Casper)))

plt.xlabel('Iterations')
plt.ylabel('Accuracy')

plt.title('Accuracy comparison in %d Iterations\n'
          'The accuracy of the DNN structure modified using\n'
          'the Casper algorithm is reduced an average of %.2f %%.'
          % (iteration, sum(result) / len(result)))

plt.legend()

plt.show()
plt.close()

print('\nA figure of comparison result has been generated')

print('\nThe code has been run %d times.\n'
      'The accuracy of the DNN structure modified using the Casper algorithm is reduced an average of %.2f %%.' % (
          iteration, sum(result) / len(result)))

print(
    "On average, the testing accuracy of the second training stage has decreased %.2f %% than the first stage." % (
        ((sum(testing_acc_DNN) / len(testing_acc_DNN)) - (
                sum(testing_acc_Casper) / len(testing_acc_Casper)))))

print("Average testing accuracy of DNN used as initialization of Casper: %.2f %%" % (sum(testing_acc_DNN) / len(testing_acc_DNN)))
print("Average testing accuracy of DNN trained with full dataset: %.2f %%" % (
        sum(testing_acc_DNN_with_full_training_set) / len(testing_acc_DNN_with_full_training_set)))
print("Average testing accuracy of modified DNN with Casper algorithm: %.2f %%" % (
        sum(testing_acc_Casper) / len(testing_acc_Casper)))

print('\nAll console info has been saved into console_record.txt')
