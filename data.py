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
from torch.utils.data import Dataset


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
    self.train = None
    self.test = None
   
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

def FetalPlanes_numpy(directoryxlsx,
                      directoryimg, val_size):
    """

        :param directoryxlsx:
        :param directoryimg:
        :return: train, test data
        """
    df = pd.read_excel(directoryxlsx)

    # creating empty arrays for the desired outputs
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
        self.train = None
        self.test = None


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
    imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val = FetalPlanes_numpy(
        directoryxlsx, directoryimg, val_size)

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


def load_CheXpert(label_dir, data_dir, batch_size, split_seed=10, validation_split=None, use_og_split=True,
                  problem_type="Binary",
                  one_channel=True, sample_strategy=None):
    """
    load train and test loaders from fetal planes ds
    """
    # use these for split in data...
    if use_og_split:
        test_csv = pd.read_csv(os.path.join(label_dir, "valid.csv"), delimiter=",")
        train_csv = pd.read_csv(os.path.join(label_dir, "train.csv"), delimiter=",")
    else:
        # TODO: add splitting function
        test = ""
        train = ""

    # transforms for dataloaders #
    transforms = CXTransformations(one_channel=one_channel)
    train_transform = transforms.train
    test_transform = transforms.test
    train_set = CheXpertDataset(train_csv, data_dir, train_transform, problem_type)
    test_set = CheXpertDataset(test_csv, data_dir, test_transform, problem_type)
    num_classes = train_set.num_classes

    # split val and train
    if validation_split:
        train_set, val_set = sklearn.model_selection.train_test_split(
            train_set, shuffle=True, test_size=validation_split, random_state=split_seed)
    else:
        val_loader = None
    # make dataloader #
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader, val_loader, num_classes


class CXTransformations:
    """
    transformations for https://arxiv.org/pdf/1901.07031.pdf
    """

    def __init__(self, resize=(300, 400), crop=(224, 224),
                 normalize=((0, 0, 0), (1, 1, 1))):
        self.train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(10),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)

        ]
        )
        self.test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        )


class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_f, data_dir, transform=None, problem_type="Binary"):
        self.csv_f = csv_f
        self.data_dir = data_dir
        self.transform = transform
        if problem_type == "Binary_CX":
            # problem: finding, no finding
            self.class_key = 'No Finding'
            self.num_classes = 2

    def __len__(self):
        return len(self.csv_f)

    def __getitem__(self, index):
        filename = self.csv_f["Path"][index]
        label = self.csv_f[self.class_key][index]
        image = PIL.Image.open(path.join(self.data_dir, filename))
        if self.transform is not None:
            image = self.transform(image)

        data = {
            "image": image,
            "label": label,
        }
        return data

