"""
File contains functions for transforming image sets
Numpy Resize - Resize numpy arrange of image 
Numpy Greyscale - Turns a numpy array of an RGB image to greyscale array
"""

import skimage as ski
import numpy as np

# NP_Resize() - resizes numpy image array into uniform smaller array.
def NP_Resize(npImageIn, resizeX, resizeY, Channels):
    npImageOut = np.zeros(np.size(npImageIn, 0), resizeX, resizeY, Channels)
    for n, img in enumerate(npImageIn):
        npImageOut[n, :, :, :] = ski.resize(npImageIn[n, :, :, :], npImageOut.shape[1:], anti_aliasing=True)
    return npImageOut

def NP_GreyScale(npImageIn):
    # Image array must have uniformly sized images
    npImageOut = np.zeros(np.size(npImageIn, 0), np.size(npImageIn, 1), np.size(npImageIn, 2))
    for n, img in enumerate(npImageIn):
        red, green, blue = npImageIn[n, :, :, 0], npImageIn[n, :, :, 1], npImageIn[n, :, :, 2]
        npImageOut[n, :, :] = 0.299*red + 0.587*green + 0.114*blue # NTSC representation of colour
    return npImageOut
