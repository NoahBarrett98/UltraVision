"""
1. Loading data as npy format
2. Creating pytorch dataloaders
"""
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
import PIL
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def FetalPlanes_numpy(csv_f,
                    directoryimg, val_size):
    """

        :param directoryxlsx:
        :param directoryimg:
        :return: train, test data
        """

    df = pd.read_csv(csv_f, delimiter=";")

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
    for i, row in tqdm(df.iterrows(), desc='loading data'):
        if i ==10:
            break
        imgname = df.at[i, 'Image_name'] + '.png'
        imgloc = os.path.join(directoryimg, imgname)
        img = Image.open(imgloc)
        array = np.asarray(img)
        plane = df.at[i, 'Plane']
        output = np.asarray(plane)
        if row['Train '] == 1:
            X_train.append(array)
            y_train.append(plane)
        else:
            X_test.append(array)
            y_test.append(plane)
        imagenames.append(imgname)
        imagelocs.append(imgloc)
        imagearrays.append(array)
        planesout.append(output)
        img.close()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val

class FetalPlanesTransform:

  def __init__(self):
    self.train = T.Compose([
                            T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),  # grayscale
                            T.Grayscale(num_output_channels=3),
                            T.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5)),
                            T.RandomPerspective(),T.RandomRotation(degrees = (0,360)),
                            T.CenterCrop(size=(224,224)),T.ToTensor()
    ])
    self.test = T.Compose([
                            T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),  # grayscale
                            T.Grayscale(num_output_channels=3),
                            T.CenterCrop(size=(224,224)), T.ToTensor()
    ])
   
class FetalPlaneDataset(Dataset):
    # initalize data and import data
    def __init__(self, img_array, label_array, transform):
        self.img_array = img_array
        self.label_array = label_array
        # encoding
        self.target_transform = {label:i for i, label in enumerate(set(label_array))}
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.img_array[index]), \
               self.target_transform[self.label_array[index]]

    def __len__(self):
        return len(self.img_array)

def FetalPlanes(directoryxlsx, directoryimg, val_size, batch_size):
    imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, y_train, y_test, y_val = FetalPlanes_numpy(directoryxlsx, directoryimg, val_size)
    
    transform = FetalPlanesTransform()
    train_dataset = FetalPlaneDataset(X_train, y_train, transform.train)
    test_dataset = FetalPlaneDataset(X_test, y_test, transform.test)
    val_dataset = FetalPlaneDataset(X_val, y_val, transform.test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
    num_outputs = 6
    return train_loader, test_loader, val_loader, num_outputs

    """
    
    :return: train, test loader
    """



def CheXpert(label_dir, data_dir, batch_size, split_seed=10, validation_split=None, use_og_split=True,
                  problem_type="Binary",
                  train_strategy="train_classification" ):
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

    # T for dataloaders #
    if train_strategy=="train_classification":
        T = CXTransformations()
        train_transform = T.train
        test_transform = T.test
    elif train_strategy == "train_simclr":
        T = ContrastiveLearningViewGenerator()
        train_transform = T
        test_transform = T

    train_set = CheXpertDataset(train_csv, data_dir, train_transform, problem_type)
    test_set = CheXpertDataset(test_csv, data_dir, test_transform, problem_type)
    num_classes = train_set.num_classes

    # split val and train
    if validation_split:
        train_set, val_set = train_test_split(
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
        self.train = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize),
            T.CenterCrop(crop),
            T.Grayscale(num_output_channels=3),
            T.RandomRotation(15),
            T.RandomAffine(10),
            T.ToTensor(),
            T.Normalize(*normalize)
        ]
        )
        self.test = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize),
            T.CenterCrop(crop),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(*normalize)
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
        image = PIL.Image.open(os.path.join(self.data_dir, filename))
        if self.transform is not None:
            image = self.transform(image)

        data = {
            "image": image,
            "label": label,
        }
        return data

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class SimCLRTransformations:
    def __init__(self, size=96, s=1):
        self.color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = T.Compose([T.ToTensor(), #nparray2tensor
                                                 T.ToPILImage(), # topilimage
                                                 T.Grayscale(num_output_channels=1), # grayscale
                                                 T.Grayscale(num_output_channels=3),
                                                  T.RandomResizedCrop(size=size),
                                                  T.RandomHorizontalFlip(),
                                                  T.RandomApply([self.color_jitter], p=0.8),
                                                  T.RandomGrayscale(p=0.2),
                                                  GaussianBlur(kernel_size=int(0.1 * size)),
                                                  T.ToTensor()])


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, n_views=2):
        self.base_transform = SimCLRTransformations().transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]