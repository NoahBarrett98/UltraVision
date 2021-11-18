import torch
from sklearn.metrics import (confusion_matrix,
                            classification_report,
                            roc_auc_score)

def evaluate(model, test_loader, train_strategy):

    if train_strategy == "train_classification":
        return evaluate_classification(model, test_loader)
    if train_strategy == "train_simclr":
        return evaluate_simclr(model, test_loader)


def evaluate_classification(model, test_loader):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long).cuda()
    pred_probs = torch.tensor([]).cuda()

    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for X, y in test_loader:
            inputs = X.cuda()
            labels = y.cuda()

            outputs = model(inputs)
            y_true = torch.cat((y_true, labels), 0)
            pred_probs = torch.cat((pred_probs, outputs), 0)

    # compute predicitions form probs
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(pred_probs, 1)

    y_pred = y_pred.cpu().numpy()
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu().numpy()
    # get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    # macro auc score
    report["auc"] = roc_auc_score(y_true, pred_probs, multi_class="ovo",average="macro")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_true, y_pred))

    
    return report

def evaluate_simclr(model, test_loader):
    # to be implemented
    raise NotImplementedError





