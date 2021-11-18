import click
import torch
import pandas as pd
import mlflow
import os
from tensorboardX import SummaryWriter
from datetime import datetime

from UltraVision import data
from UltraVision import models
from UltraVision import train
from UltraVision import evaluation
from UltraVision import optimizers
from UltraVision import utils

@click.command()
@click.option('--label_dir', default='', help='label location')
@click.option('--data_dir', default='', help='data folder')
@click.option('--data_name', default='', help='name of dataset')
@click.option('--model_name', default='', help='name of model')
@click.option('--train_strategy', default='train_simclr', help='name of training strategy')
@click.option('--use_scheduler', default=True, help='whether to use scheduler')
@click.option('--batch_size', default=8, type=int, help='batch size for dataloaders')
@click.option('--num_epochs', default=100, type=int, help='number of epochs')
@click.option('--val_size', default=0.1, type=float, help='validation split size')
@click.option('--lr', default=0.01, type=float, help='optimizer learning rate')
@click.option('--momentum', default=0.5, type=float, help='optimizer momentum')
@click.option('--optimizer_name', default='SGD', help='optimizer momentum')
@click.option('--exp_name', default=None, help='name of experiment')
@click.option('--use_tensorboard', default=True,type=bool, help='wether to use tensorboard or not')
def train_model(label_dir, data_dir, data_name, model_name, train_strategy, use_scheduler, batch_size, val_size, num_epochs,
        lr, momentum, optimizer_name, exp_name, use_tensorboard):

    # start mlflow experiment
    mlflow.set_experiment(exp_name)

    # setup tensorboard
    if use_tensorboard:
        if not os.path.exists("tensorboard"):
            os.makedirs("tensorboard")
        tensorboard_location = os.path.join("tensorboard", f'{exp_name}_{datetime.now().strftime("%Y%m%d_%H:%M")}')
        writer = SummaryWriter(tensorboard_location)
    else:
        tensorboard_location = "None"
        writer = None

    # load data
    train_loader, test_loader, val_loader, num_outputs = data.__dict__[data_name](label_dir, data_dir,
                                                                                  val_size, batch_size)
    # load model
    model = models.__dict__[model_name](pretrained=True, num_outputs=num_outputs)
    model = model.cuda()
    # get optimizer
    optimizer = optimizers.__dict__[optimizer_name](model, lr, momentum)

    # set scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
    else:
        scheduler = None

    # train model
    model = train.__dict__[train_strategy](model=model, optimizer=optimizer,
                                        scheduler=scheduler,  train_loader=train_loader,
                                           val_loader=val_loader, num_epochs=num_epochs,
                                           writer=writer
                                           , num_outputs=num_outputs)
    # evaluate model
    eval_results = evaluation.evaluate(model, test_loader, train_strategy)

    with mlflow.start_run():
        mlflow.log_param("data_name", data_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("train_strategy", train_strategy)
        mlflow.log_param("use_scheduler", use_scheduler)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("tboard loc", tensorboard_location)
        mlflow.log_metric("auc", eval_results["auc"])
        mlflow.log_metric("accuracy", eval_results["accuracy"])

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
