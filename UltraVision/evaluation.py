def evaluate(model, test_loader, train_strategy):

    if train_strategy == "train_classifcation":
        evaluate_classification(model, test_loader)
    if train_strategy == "train_simclr":
        evaluate_simclr(model, test_loader)


def evaluate_classification(model, test_loader):
    # to be implemented
    model.eval()
def evaluate_simclr(model, test_loader):
    # to be implemented
    model.eval()





