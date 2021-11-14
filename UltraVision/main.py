import click
import torch
import pandas as p

from UltraVision import data
from UltraVision import models
from UltraVision import train
from UltraVision import evaluation
from UltraVision import optimizers

@click.command()
@click.option('--label_dir', default='', help='label location')
@click.option('--data_dir', default='', help='data folder')
@click.option('--data_name', default='', help='name of dataset')
@click.option('--model', default='', help='name of model')
@click.option('--train_strategy', default='train_simclr', help='name of training strategy')
@click.option('--use_scheduler', default=True, help='whether to use scheduler')
@click.option('--batch_size', default=8, type=int, help='batch size for dataloaders')
@click.option('--num_epochs', default=100, type=int, help='number of epochs')
@click.option('--val_size', default=0.1, type=float, help='validation split size')
@click.option('--lr', default=0.01, type=float, help='optimizer learning rate')
@click.option('--momentum', default=0.5, type=float, help='optimizer momentum')
@click.option('--optimizer', default='SGD', help='optimizer momentum')
def train_model(label_dir, data_dir, data_name, model, train_strategy, use_scheduler, batch_size, val_size, num_epochs,
        lr, momentum, optimizer):

    # load data
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](label_dir, data_dir,
                                                                                  val_size, batch_size)

    # load model
    model = models.__dict__[model](num_outputs)
    model = model.cuda()
    # get optimizer
    optimizer = optimizers.__dict__[optimizer](model, lr, momentum)

    # set scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
    else:
        scheduler = None

    # train model
    model = train.__dict__[train_strategy](model=model, optimizer=optimizer,
                                        scheduler=scheduler,  train_loader=train_loader,
                                           val_loader=val_loader, num_epochs=num_epochs,
                                           writer=None, num_outputs=num_outputs)
    # evaluate model
    eval_results = evaluation.evaluate(model, test_loader, train_strategy)
    return model

@click.command()
@click.option('--label_dir', default='', help='label location')
@click.option('--data_dir', default='', help='data folder')
@click.option('--data_name', default='', help='name of dataset')
@click.option('--testSize', default=0.1, type=float, help='percentage to be used for test set')
@click.option('--resizeX', default=224, type=int, help='SHAPE OF X RESIZE')
@click.option('--resizeY', default=224, type=int, help='shape of y resize')
def runSciLearn(label_dir, data_dir, data_name, testSize, resizeX, resizeY):
    channels = 3  # RGB
    imagenames, imagelocs, imagearrays, planesout, X_test, X_train, X_val, Y_train, Y_test, Y_val = data.FetalPlanes_numpy(
        label_dir, data_dir, testSize)

    X_train = CustomTransforms.NP_Resize(X_train, resizeX, resizeY, channels)
    X_test = CustomTransforms.NP_Resize(X_test, resizeX, resizeY, channels)
    Y_train = CustomTransforms.NP_Resize(Y_train, resizeX, resizeY, channels)
    Y_test = CustomTransforms.NP_Resize(Y_test, resizeX, resizeY, channels)

    X_train = CustomTransforms.NP_GrayScale(X_train)
    X_test = CustomTransforms.NP_GrayScale(X_test)
    Y_train = CustomTransforms.NP_GrayScale(Y_train)
    Y_test = CustomTransforms.NP_GrayScale(Y_test)

    knn = models.KNN(X_train, Y_train)
    svc = models.SVC(X_train, Y_train)
    gnb = models.GNaiveBayes(X_train, Y_train)

    knnClassReport = p.dataFrame(models.PredictAndClass(knn, X_test, Y_test))
    svcClassReport = p.dataFrame(models.PredictAndClass(svc, X_test, Y_test))
    gnbClassReport = p.dataFrame(models.PredictAndClass(gnb, X_test, Y_test))

    pass

# if __name__ == "__main__":
#     run()