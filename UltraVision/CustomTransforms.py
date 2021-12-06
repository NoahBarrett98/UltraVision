"""
File contains functions for transforming image sets
Numpy Resize - Resize numpy arrange of image 
Numpy Greyscale - Turns a numpy array of an RGB image to greyscale array
NP_LineImage - Converts greyscale image array into dimension by (x*y) 2d array
hogTransform - Converts a greyscale numpy representation of an image into a Histogram of Gradients
"""

import skimage.transform as ski
import numpy as np
from skimage.feature import hog

def NP_Resize(npImageIn, resizeX, resizeY, Channels):
    print("Resizing")
    npImageOut = np.full((np.size(npImageIn, 0), resizeX, resizeY, Channels), 0, dtype=float)
    for n, img in enumerate(npImageIn):
        tempImg = npImageIn[n]
        resImg = ski.resize(tempImg, (resizeX, resizeY, Channels), anti_aliasing=True)
        npImageOut[n, :, :, :] = resImg
    return npImageOut

def NP_GreyScale(npImageIn):
    # Image array must have uniformly sized images
    print("Greyscale Convert")
    npImageOut = np.full((np.size(npImageIn, 0), np.size(npImageIn, 1), np.size(npImageIn, 2)), 0, dtype=float)
    for n, img in enumerate(npImageIn):
        red, green, blue = npImageIn[n, :, :, 0], npImageIn[n, :, :, 1], npImageIn[n, :, :, 2]
        npImageOut[n, :, :] = 0.299*red + 0.587*green + 0.114*blue # NTSC representation of colour
    return npImageOut


def NP_LineImage(npImageIn):
    print("Reshaping")
    nCount, nx, ny = npImageIn.shape
    npImageOut = npImageIn.reshape((nCount, nx*ny))
    return npImageOut

def hogTransform(npImageIn):
    print("Hog Transforming")
    npImageOut = []
    for n, img in enumerate(npImageIn):
        tempImg = npImageIn[n, :, :]
        flatHog = hog(tempImg, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
        npImageOut.append(flatHog)
    return npImageOut