# This script creates a dataloader for Pytorch

import torchvision as tv
import numpy as np
import pandas as pd
import torch
import category_encoders as ce
import matplotlib.pyplot as plt
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class FetalPlaneDataset(Dataset):

    # initalize data and import data
    def __init__(self):
        # finding the image and xlsx directories
        directorycsv = r"C:\Users\conno\ML_Scripts\ML_Project\FETAL_PLANES_ZENODO"
        directoryimg = r"C:\Users\conno\ML_Scripts\ML_Project\FETAL_PLANES_ZENODO\Images"
        xlsx = "FETAL_PLANES_DB_data.xlsx"
        xlsxloc = os.path.join(directorycsv, xlsx)

        # creating a dataframe based on xlsx data
        df = pd.read_excel(xlsxloc)

        # declaring empty arrays to store image data (features) as the corresponding plane (output)
        imagenames = []
        imagelocs = []
        imagearrays = []
        planesout = []

        # This loop is iterating through the dataframe to find the correct image and plane output then store the image data
        # numpy array in another array (imagearrays). Also storing the corresponding plane to the planesout array.
        for i, rows in df.iterrows():
            imgname = df.at[i, 'Image_name'] + '.png'
            imgloc = os.path.join(directoryimg, imgname)
            img = Image.open(imgloc)
            array = np.asarray(img)
            plane = df.at[i, 'Plane']
            output = np.asarray(plane)
            imagenames.append(imgname)
            imagelocs.append(imgloc)
            imagearrays.append(array)
            planesout.append(output)
            img.close()

        self.output_data = torch.as_tensor(planesout)
        self.feat_data = torch.as_tensor(imagearrays)
        self.len = planesout.shape[0]

    def __getitem__(self, index):
        return self.feat_data[index], self.output_data[index]

    def __len__(self):
        return self.len

print('dont')