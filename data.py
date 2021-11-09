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
from torchvision import transforms
import transforms as T

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

class FetalPlanesTransform:

  def __init__(self):
    self.train = torchvision.transform.Compose(T.Greyscale(), T.GaussianBlur(kernel_size=(5,9), sigma(0.1,5)),
                                               T.RandomPerspective(),T.RandomRotation(degrees = (0,360)),
                                               T.CenterCrop(size=(224,224)),T.ToTensor())
    self.test = torchvision.transform.Compose(T.Greyscale(), T.CenterCrop(size=(224,224)), T.ToTensor())
   
class FetalPlaneDataset(Dataset):
    # initalize data and import data
    def __init__(self, img_array, label_array):
        self.img_array = img_array
        self.label_array = label_array
        self.len = label_array.shape[0]

    def __getitem__(self, index):
        return transform(self.img_array[index]), self.label_array[index]

    def __len__(self):
        return self.len

def FetalPlanes(directoryxlsx, directoryimg, val_size, batch_size):
    imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val = FetalPlanes_numpy(directoryxlsx, directoryimg, val_size)
    
    transform = FetalPlanesTransform()
    train_dataset = FetalPlaneDataset(X_train, y_train, transform.train)
    test_dataset = FetalPlaneDataset(X_test, y_test, transform.test)
    val_dataset = FetalPlaneDataset(X_val, y_val, transform.test)

    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size)
    test_loader = torch.utils.data.Dataloader(test_dataset, batch_size)
    val_loader = torch.utils.data.Dataloader(val_dataset, batch_size)

    return train_loader, test_loader, val_loader

    """
    
    :return: train, test loader
    """

