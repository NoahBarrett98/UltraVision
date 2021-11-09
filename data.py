"""
1. Loading data as npy format
2. Creating pytorch dataloaders
"""
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def FetalPlanes_numpy(directoryxlsx,
                    directoryimg, val_size):
    """

        :param directoryxlsx:
        :param directoryimg:
        :return: train, test data
        """
    df = pd.read_excel(directoryxlsx)

    #creating empty arrays for the desired outputs
    imagenames = []
    imagelocs = []
    imagearrays = []
    planesout = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []

# This section is iterating through the imiage data to convert it into numerical data and creating the train, test, val
# datasets to be used in our models
    for i, rows in df.iterrows():
        imgname = df.at[i, 'Image_name'] + '.png'
        imgloc = os.path.join(directoryimg, imgname)
        img = Image.open(imgloc)
        array = np.asarray(img)
        plane = df.at[i, 'Plane']
        output = np.asarray(plane)
        if df['Train'] == 1:
            X_train.append(array)
            y_train.append(output)
        else:
            X_test.append(array)
            y_test.append(array)
        imagenames.append(imgname)
        imagelocs.append(imgloc)
        imagearrays.append(array)
        planesout.append(output)
        img.close()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val

class FetalPlaneDataset(Dataset):
    # initalize data and import data
    def __init__(self, img_array, label_array):
        # FetalPlanes_numpy(directoryxlsx, directoryimg, val_size)
        # self.output_data = torch.as_tensor(planesout)
        # self.feat_data = torch.as_tensor(imagearrays)
        self.img_array = img_array
        self.label_array = label_array
        self.len = label_array.shape[0]

    def __getitem__(self, index):
        return self.img_array[index], self.label_array[index]

    def __len__(self):
        return self.len

def FetalPlanes(directoryxlsx, directoryimg, val_size, batch_size):
    imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val = FetalPlanes_numpy(directoryxlsx, directoryimg, val_size)
    train_dataset = FetalPlaneDataset(X_train, y_train)
    test_dataset = FetalPlaneDataset(X_test, y_test)
    val_dataset = FetalPlaneDataset(X_val, y_val)

    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size)
    test_loader = torch.utils.data.Dataloader(test_dataset, batch_size)
    val_loader = torch.utils.data.Dataloader(val_dataset, batch_size)

    return train_loader, test_loader, val_loader

    """
    
    :return: train, test loader
    """

