import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt


def train(net, train_loader, val_loader, epochs, optimizer, loss, device):

    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs],
                            legend=['train loss', 'train acc', 'val acc'])
    timer, num_batches = d2l.Timer(), len(train_loader)
    for epoch in range(epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_loader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        val_acc = d2l.evaluate_accuracy_gpu(net, val_loader)
        animator.add(epoch + 1, (None, None, val_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'val acc {val_acc:.3f}')
    print(f'{metric[2] * epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    
    


