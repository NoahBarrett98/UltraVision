import click
import torch

import data
import models
import train
import evaluation
import optimizers

@click.command()
@click.option('--data_dir', default='', help='data folder')
@click.option('--data_name', default='', help='name of dataset')
@click.option('--model', default='', help='name of model')
@click.option('--train_strategy', default='train_simclr', help='name of training strategy')
@click.option('--use_scheduler', default=True, help='whether to use scheduler')
@click.option('--batch_size', default=8, help='batch size for dataloaders')
@click.option('--num_epochs', default=100, help='number of epochs')
@click.option('--lr', default=0.01, help='optimizer learning rate')
@click.option('--momentum', default=0.5, help='optimizer momentum')
def run(data_dir, data_name, model, train_strategy, use_scheduler, criterion, batch_size, num_epochs,
        lr, momentum):

    # load data
    train_loader, test_loader = data.__dict__[data_name](data_dir, batch_size)

    # load model
    model = models.__dict__[model]()

    # get optimizer
    optimizer = optimizers.__dict__[optimizers](model.params, lr, momentum)

    # set scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
    else:
        scheduler = None

    # train model
    model = train.__dict__[train_strategy](model, optimizer, scheduler, criterion, train_loader, num_epochs)

    # evaluate model
    eval_results = evaluation.evaluate(model, test_loader)
    pass

if __name__ == "__main__":
    run()
