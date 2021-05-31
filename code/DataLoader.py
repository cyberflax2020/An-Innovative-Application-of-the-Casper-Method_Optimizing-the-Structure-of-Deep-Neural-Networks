import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Dataloader:

    # link to the data source
    def __init__(self, source_path=r'Anger.xlsx'):
        super(Dataloader, self).__init__()
        self.source_path = source_path

        self.train_input = None
        self.train_target = None
        self.vali_input = None
        self.vali_target = None
        self.test_input = None
        self.test_target = None
        self.train_larger_input = None
        self.train_larger_target = None
        self.train_smaller_input = None
        self.train_smaller_target = None

    # preprocess the data and return datasets for training, validation and testing
    def load_and_preprocess(self):
        # load in data from .xlsx file
        data = pd.read_excel(self.source_path, sheet_name=0, header=0)

        # delete the redundant columns, leaving only pure data, and set column names
        coln = list(data)
        coln[0] = "0"
        data.columns = coln
        data = data.drop(labels="0", axis=1)
        # note: dont use observer information for training
        data = data.drop(labels="Video", axis=1)

        # Labelling the labels using 0 or 1
        le = preprocessing.LabelEncoder()
        le.fit(data["Label"].values.tolist())
        integer_mapping = {l: i for i, l in enumerate(le.classes_)}
        data["Label"] = le.transform(data["Label"].values.tolist())

        # check the labels
        print("\nThe labels have been set as: ", integer_mapping)

        # convert object to numeric
        data = data._convert(numeric=True)

        # data normalization
        for i in ["Mean", "Std", "Diff1", "Diff2", "PCAd1", "PCAd2"]:
            data[i] = (data[i] - data[i].mean()) / data[i].std()

        # train, validation, test set split

        # the 1st approach: using train_test_split with a random state (We don't use this)

        # df_train, df_test_vali = train_test_split(data, train_size=0.8)
        # df_vali, df_test = train_test_split(df_test_vali, train_size=0.5)

        # the 2nd approach: construction balanced datasets manually (Use this one)

        # extract all data with the same label:
        df_G = data.loc[data['Label'] == 1].sample(frac=1).reset_index(drop=True)
        df_F = data.loc[data['Label'] == 0].sample(frac=1).reset_index(drop=True)

        # reconstruct the dataset and let the two labels alternate
        data = data.drop(index=data.index)
        for i in range(len(df_G)):
            data.loc[2 * i] = df_G.loc[i]
            data.loc[2 * i + 1] = df_F.loc[i]

        # set params: tr_vte for train size/(vali size + test size), v_te for vali size / test size
        tr_vte = 0.6
        v_te = 0.8
        data_size = len(data)

        # preform the split, note that only in train set, keep the labels alternate
        df_train = data.loc[0:tr_vte * data_size - 1].reset_index(drop=True)
        df_vali = data.loc[tr_vte * data_size:v_te * data_size - 1].sample(frac=1, random_state=None).reset_index(
            drop=True)
        df_test = data.loc[v_te * data_size:data_size].sample(frac=1, random_state=None).reset_index(drop=True)

        # train set:
        n_features = df_train.shape[1] - 1
        train_input = torch.Tensor((df_train.iloc[:, :n_features]).values).float()
        train_target = torch.Tensor((df_train.iloc[:, n_features]).values).long()

        # validation set:
        vali_input = torch.Tensor(df_vali.iloc[:, :n_features].values).float()
        vali_target = torch.Tensor(df_vali.iloc[:, n_features].values).long()

        # test set:
        test_input = torch.Tensor(df_test.iloc[:, :n_features].values).float()
        test_target = torch.Tensor(df_test.iloc[:, n_features].values).long()

        # further divide train set into 2 parts for future use
        larger_size = 160


        df_train_larger = df_train.loc[0:larger_size - 1].reset_index(drop=True)
        df_train_smaller = df_train.loc[larger_size:len(df_train)].reset_index(drop=True)

        train_larger_input = torch.Tensor((df_train_larger.iloc[:, :n_features]).values).float()
        train_larger_target = torch.Tensor((df_train_larger.iloc[:, n_features]).values).long()

        train_smaller_input = torch.Tensor((df_train_smaller.iloc[:, :n_features]).values).float()
        train_smaller_target = torch.Tensor((df_train_smaller.iloc[:, n_features]).values).long()

        self.train_input = train_input
        self.train_target = train_target
        self.vali_input = vali_input
        self.vali_target = vali_target
        self.test_input = test_input
        self.test_target = test_target
        self.train_larger_input = train_larger_input
        self.train_larger_target = train_larger_target
        self.train_smaller_input = train_smaller_input
        self.train_smaller_target = train_smaller_target

    # This part is inspired by the ANU lab code, it is used to plot a confusion matrix.
    @staticmethod
    def plot_confusion(input_sample, num_classes, des_output, actual_output):
        confusion = torch.zeros(num_classes, num_classes)
        for i in range(input_sample):
            actual_class = actual_output[i]
            predicted_class = des_output[i]

            confusion[actual_class][predicted_class] += 1

        return confusion
