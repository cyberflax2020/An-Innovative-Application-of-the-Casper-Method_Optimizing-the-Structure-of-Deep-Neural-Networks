Thank you for reading my paper!

Before running my code, please make sure you have updated to the latest version of Pytorch and Openpyxl, otherwise you may not be able to read the data set.

Please make sure that the following packages have been added:

import os
import sys
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd
import torch
from sklearn import preprocessing

Please make sure to give your Pycharm administrator permissions, otherwise this project may not be able to read and write on your disk.

The main script of this project is main.py. It will train, validate, and test the following three models in sequence:

1. The first stage of DNN-Casper composite network
2. The DNN using the complete data set
3. The second stage of DNN-Casper composite network

You may also run get_average_result_comparison.py to automatically run this script repeatedly to obtain the comparison between the results of multiple runs and generate a graph of averaged results. But please note that this will be very time consuming.

After the running, you can read the saved  results in the console_record.txt which will be generated in the same directory.

Thanks again.