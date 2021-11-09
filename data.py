"""
1. Loading data as npy format
2. Creating pytorch dataloaders
"""


def FetalPlanes_numpy(directoryxlsx,
                    directoryimg):
    """

        :param directoryxlsx:
        :param directoryimg:
        :return: train, test data
        """
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

def FetalPlanes():
    """

    :return: train, test loader
    """

