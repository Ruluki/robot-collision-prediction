import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # STUDENTS: it may be helpful for the final part to balance the distribution of
        # your collected data
        self.data = self.data[~np.isnan(self.data).any(axis=1)]
        class0 = self.data[self.data[:,-1] == 0]
        class1 = self.data[self.data[:,-1] == 1]
        if len(class0) > len(class1):
            # Calculate the number of samples to keep, target 5:1
            num_samples_to_keep = len(class1)
            # Randomly select samples from class0 to keep
            class0_indices_to_keep = np.random.choice(len(class0),size=num_samples_to_keep, replace=False)
            class0_balanced = class0[class0_indices_to_keep]
            self.data = np.concatenate((class0_balanced, class1), axis=0)
        # Normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)  # fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))  # save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # STUDENTS: for this example, __getitem__() must return a dict with entries
        # {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but
        # to work with autograding, please do not deviate from these specifications.
        x = torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.normalized_data[idx, -1], dtype=torch.float32)
        return {'input': x, 'label': y}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        # STUDENTS: randomly split dataset into two DataLoaders, self.train_loader and
        # self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as
        # this may vary
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.nav_dataset, [train_size, test_size])
        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    train_class_counts = {0: 0, 1: 0}
    test_class_counts = {0: 0, 1: 0}
    # STUDENTS: note this is how the dataloaders will be iterated over, and cannot
    # be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, labels = sample['input'], sample['label']
        for label in labels:
            train_class_counts[int(label.item())] += 1
    for idx, sample in enumerate(data_loaders.test_loader):
        _, labels = sample['input'], sample['label']
        for label in labels:
            test_class_counts[int(label.item())] += 1
    print("Train dataset class counts:", train_class_counts)
    print("Test dataset class counts:", test_class_counts)
    print(data_loaders.nav_dataset.__len__())
    for i in range(6):
        print(data_loaders.nav_dataset.__getitem__(i))

if __name__ == '__main__':
    main()
