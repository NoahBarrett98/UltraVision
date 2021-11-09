import torchvision.models as models
from torch import nn
import torch
import torch.nn.functional as F

class HighResCNN(nn.Module):
    """
    model built to read in full sized images
    """
    def __init__(self):
        super(HighResCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size= (3,3), stride=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv11 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.avgpool(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2):
        super(DenseNet169, self).__init__()
        self.dn = rue
        self.base = self.dn.features
        self.classifier = torch.nn.Linear(in_features=self.dn.classifier.in_features, out_features=num_outputs)

    def forward(self, x):
        x = self.base(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained, num_outputs=2):
        super(ResNet18, self).__init__()
        self.base = models.resnet18(pretrained=pretrained)
        self.conv1 = self.base.conv1
        self.bn1 = self.base.bn1
        self.relu = self.base.relu
        self.maxpool = self.base.maxpool
        self.layer1 = self.base.layer1
        self.layer2 = self.base.layer2
        self.layer3 = self.base.layer3
        self.layer4 = self.base.layer4
        self.avgpool = self.base.avgpool
        self.__in_features = self.base.fc.in_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=num_outputs, bias=True),
        )

    def forward(self, x):
        x = self.feature_forward(x)
        output = self.classifier(x)
        return output

    def feature_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def feature_dim(self):
        return self.__in_features

"""
models.py

Contains the different methods for classification

KNeighbors Classification               (Not Implemented)
Linear SVC/SVM                          (Not Implemented)
SVC                                     (Not Implemented)
HOG-SVM - Stochastic Gradient Descent   (Not Implemented)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# KNearestNeighbors
def KNN():
    knn = KNeighborsClassifier()
    return knn

# SVC
# Images for SVC/SVM must be the same dimensions
def SVC():
    svc = svm.SVC(probability=True)
    return svc

#Gaussian Naive Bayes
def GNaiveBayes():
    gnb = GaussianNB()
    return gnb
