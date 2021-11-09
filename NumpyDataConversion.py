# This program converts visual data into numpy arrays

from PIL import Image
import numpy as np
import pandas as pd
import os
 # = r"C:\Users\conno\ML_Scripts\ML_Project\FETAL_PLANES_ZENODO\FETAL_PLANES_DB_data.xlsx"
#  = r"C:\Users\conno\ML_Scripts\ML_Project\FETAL_PLANES_ZENODO\Images"
def NumpyConversion(directoryxlsx,
                    directoryimg):


    df = pd.read_excel(directoryxlsx)

    imagenames = []
    imagelocs = []
    imagearrays = []
    planesout = []

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

    return imagenames, imagelocs, imagearrays, planesout
