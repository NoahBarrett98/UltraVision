
def evaluate(model, test_loader, train_strategy):

    if train_strategy == "train_simclr":
        evaluate_simclr(model, test_loader)

def evaluate_simclr(model, test_loader):
    model.eval()





