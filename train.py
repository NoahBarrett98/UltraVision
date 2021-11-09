import torch
import tqdm
import evaluation

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

def train_classification(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs,
                         num_outputs):
    pbar = tqdm.tqdm(range(num_epochs))

    # training
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['image'].cuda(), data['label'].cuda()
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
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data['image'].cuda(), data['label'].cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        pbar.set_description(f"Loss: {running_loss / len(train_loader):.6f}")
        print(f"running loss: {running_loss}")
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            train_results = evaluation.evaluate(model, train_loader, verbose=False, num_outputs=num_outputs)
            val_results = evaluation.evaluate(model, val_loader, verbose=False, num_outputs=num_outputs)
    return model
