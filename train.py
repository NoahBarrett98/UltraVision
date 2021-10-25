import torch
import tqdm

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
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
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