from torch import optim


def SGD(model, lr, momentum):
    return optim.SGD([
        {'params': model.base.parameters(), 'lr':lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], momentum=momentum)

def Adam(model, lr, weight_decay):
    return optim.Adam([
        {'params': model.base.parameters(), 'lr':lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=weight_decay)