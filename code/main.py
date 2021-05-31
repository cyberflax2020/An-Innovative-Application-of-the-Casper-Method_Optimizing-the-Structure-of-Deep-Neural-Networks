import os
import sys

from DataLoader import Dataloader
from Logger import Logger
from DNN_train_evaluation import DNN_trainer
from DNN_train_evaluation import DNN_trainer_with_full_training_set
from Casper_train_evaluation import Casper_trainer

'''
Author: Yuliang Ma (u6462980)
Dataset: Anger
Approach: Casper, DNN
'''

'''
This is the main script of this project.
You may also run get_average_result_comparison.py to 
automatically run this script repeatedly to obtain 
the comparison between the results of multiple runs 
and generate a graph of averaged results.
'''

try:
    os.remove('console_record.txt')
except BaseException as e:
    print(e)
    print("\nCreated new files")

# this part is a simple report generator which saves all console info in to a .txt file
sys.stdout = Logger("console_record.txt")
print("\nRunning main.py")

# load and preprocess the data
dataloader = Dataloader(r'Anger.xlsx')
dataloader.load_and_preprocess()

'''
Run this part of the code several times until you get a relatively high accuracy DNN model
dnn_trainer.test() will save the trained DNN in DNN_model.pkl
After getting the desired DNN model,
you may comment out the following three lines to keep the model.
All key hyper parameter settings can be adjusted in the constructor of the DNN_trainer class.
This part uses 2/3 of the training set.
'''

dnn_trainer = DNN_trainer(dataloader)
dnn_trainer.train()
dnn_trainer.test()

'''
This part below is basically the same with the above.
It trains a classical DNN with full training dataset for comparison 
'''

dnn_trainer_with_full_training_set = DNN_trainer_with_full_training_set(dataloader)
dnn_trainer_with_full_training_set.train()
dnn_trainer_with_full_training_set.test()

'''
# Run the following three lines to use the DNN model obtained above
# as the initial structure of the Casper algorithm.
# The DNN model modified by Casper will be saved in Casper_model.pkl.
# test_comparison() will compare the results of the two models at the end.
# All key hyper parameter settings can be adjusted in the constructor of the Casper_trainer class.
This part uses rest of the 1/3 of the training set.
'''

casper_trainer = Casper_trainer(dataloader)
casper_trainer.train_validate()
casper_trainer.test_comparison()


print('\nAll console info has been saved into console_record.txt')