import torch
import pandas as pd


class irisDataSet(torch.utils.data.Dataset):
    def __init__(self, fileName, traingDataPercents):
        torch.manual_seed(1234)
        data = pd.read_csv(fileName)
        data = data.sample(frac=1,random_state=1234)
        rowCount = len(data)
        startRow = 1
        trainRowCount = int(rowCount*traingDataPercents/100)
        testRowCount = trainRowCount + int(rowCount*(100-traingDataPercents)/100)
        data['Species']=pd.Categorical(data['Species']).codes

        self.train_data = data.values[startRow:trainRowCount, 1:5]
        self.train_target = data.values[startRow:trainRowCount, 5]        
        self.test_data = data.values[trainRowCount:testRowCount, 1:5]
        self.test_target = data.values[trainRowCount:testRowCount, 5]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        train_data = self.train_data[index]
        train_target = self.train_target[index]        
        test_data = self.test_data[index]
        test_target = self.test_target[index]
        return (train_data, train_target, test_data, test_target)
