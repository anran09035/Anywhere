# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Data loading and transformation
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)
def data_transformation(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape(-1)
    x = torch.from_numpy(x)
    return x
train_set = mnist.MNIST('./data', train=True, transform=data_transformation, download=False)
test_set = mnist.MNIST('./data', train=False, transform=data_transformation, download=False)

# Data iterator
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# Construction of network model
net = nn.Sequential(
    nn.Linear(784, 40),
    nn.Tanh(),
    nn.Linear(40, 20),
    nn.Tanh(),
    nn.Linear(20, 10),
)
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

# Training and testing
overall_train_loss = []
overall_train_acc = []
overall_test_loss = []
overall_test_acc = []

start = time.time()
for i in range(10):
    train_loss = 0
    train_acc = 0
    net.train()
    # training
    for image, label in train_data:
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        out = net(image)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct.to(torch.float32) / image.shape[0]
        train_acc += acc

    overall_train_loss.append(train_loss / len(train_data))
    overall_train_acc.append(train_acc / len(train_data))

    # testing
    test_loss = 0
    test_acc = 0
    net.eval()
    for image, label in test_data:
        image = Variable(image.cuda())
        label = Variable(label.cuda())
        out = net(image)
        loss = criterion(out, label)
        test_loss += loss.data
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct.to(torch.float32) / image.shape[0]
        test_acc += acc
    overall_test_loss.append(test_loss / len(test_data))
    overall_test_acc.append(test_acc / len(test_data))

    # Output result
    during = time.time() - start
    print('Epoch: {}, Using time {:.1f}, Train Loss: {:.3f}, Train Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}'
          .format(i + 1, during, train_loss / len(train_data), train_acc / len(train_data),
                  test_loss / len(test_data), test_acc / len(test_data)))

# Draw a loss curve
plt.subplot(2, 2, 1)
plt.plot(np.arange(len(overall_train_loss)), overall_train_loss)
plt.title('overall_train_loss')

plt.subplot(2, 2, 2)
plt.plot(np.arange(len(overall_train_acc)), overall_train_acc)
plt.title('overall_train_acc')

plt.subplot(2, 2, 3)
plt.plot(np.arange(len(overall_test_loss)), overall_test_loss)
plt.title('test loss')

plt.subplot(2, 2, 4)
plt.plot(np.arange(len(overall_test_acc)), overall_test_acc)
plt.title('test acc')

plt.show()
