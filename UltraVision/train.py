import torch
import tqdm

from UltraVision import evaluation

def train_simclr(model, optimizer, scheduler, criterion, train_loader, num_epochs):
    """
    train a model using the framework proposed in
    https://arxiv.org/pdf/2002.05709.pdf

    """

    pbar = tqdm.tqdm(range(num_epochs))
    model.train()
    # training
    for epoch in pbar:
        running_loss = 0.0
        for pos_1, pos_2, target in train_loader:
            pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
            feature_i, z_i = model(pos_1)
            feature_j, z_j = model(pos_2)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if scheduler:
                scheduler.step()

            pbar.set_description(
                f"Train loss: {running_loss / len(train_loader):.3f}")

        return model

def train_classification(model, optimizer,
                        scheduler,  train_loader,
                           val_loader, num_epochs,
                           writer, num_outputs):
    pbar = tqdm.tqdm(range(num_epochs))
    # CE for classification
    criterion = torch.nn.CrossEntropyLoss()
    # training
    for epoch in pbar:
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # put on gpu
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        # get validation loss #
        with torch.no_grad():
            val_running_loss = 0.0
            for i, (inputs, labels) in enumerate(val_loader, 0):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        pbar.set_description(f"Loss: {running_loss / len(train_loader):.6f}")
        print(f"running loss: {running_loss}")
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        train_results = evaluation.evaluate(model, train_loader, train_strategy = "train_classification")
        val_results = evaluation.evaluate(model, val_loader, train_strategy = "train_classification")

        if writer:
            writer.add_scalar('loss/train', running_loss / len(train_loader), epoch)
            writer.add_scalar('accuracy/train', train_results['accuracy'], epoch)
            writer.add_scalar('auc/train', train_results['auc'], epoch)
            writer.add_scalar('loss/val', val_running_loss / len(val_loader), epoch)
            writer.add_scalar('accuracy/val', val_results['accuracy'], epoch)
            writer.add_scalar('auc/val', val_results['auc'], epoch)

    return model
