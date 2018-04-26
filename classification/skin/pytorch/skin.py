'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils.functions import progress_bar
import os
from tensorboardX import SummaryWriter

from classification.models.pytorch.vgg import VGG
from torch.autograd import Variable
from classification.skin.pytorch.dataset_isic import ISIC2017Dataset


batch_size_train_per_gpu = 50
epochs = 200
num_classes = 2
learning_rate = 0.001
log_dir='logs'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

trainset=ISIC2017Dataset(task=1, mode='train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train_per_gpu, shuffle=True, num_workers=2)

testset = ISIC2017Dataset(task=1, mode='test')
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

# Model
try:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
except Exception as e:
    print('==> Building model..')
    net = VGG('VGG19',num_classes)
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    #net = ResNeXt29_2x64d(num_classes=2)
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\n Training Epoch:{} '.format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        step = epoch * len(trainloader) + batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        writer.add_scalar('train loss',train_loss,step)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def save_model(acc, epoch):
    print('\n Saving new model with accuracy {}'.format(acc))
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7')

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    target_all=[]
    predicted_all=[]
    print("\ntesting with previous accuracy {}".format(best_acc))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        predicted_batch=predicted.eq(targets.data).cpu()
        predicted_reshaped=predicted_batch.numpy().reshape(-1)
        predicted_all=np.concatenate((predicted_all,predicted_reshaped),axis=0)

        targets_reshaped = targets.data.cpu().numpy().reshape(-1)
        target_all = np.concatenate((target_all, targets_reshaped), axis=0)
        total += targets.size(0)
        correct += predicted_batch.sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('test accuracy', acc, epoch)
    if acc > best_acc:
        save_model(acc, epoch)
        best_acc = acc

    accuracy = metrics.accuracy_score(target_all, predicted_all)

    print(accuracy)

    """
    total sum of confusion matrix value is same as total number items in test set.
    """
    cm = metrics.confusion_matrix(target_all, predicted_all)
    print(cm)

    auc = metrics.roc_auc_score(target_all, predicted_all)
    print("Auc {}".format(auc))
    writer.add_scalar('test auc', auc, epoch)


    # f1_score = metrics.f1_score(y_true, y_pred)

    # print(f1_score)

    # average_precision = metrics.average_precision_score(y_true, y_pred)
    #
    # print(average_precision)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    #print(TP)
    #print(TN)
    #
    # # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # print(ACC.mean())


try:
  for epoch in range(start_epoch, start_epoch + epochs):
      train(epoch)
      test(epoch)
except KeyboardInterrupt:
    test(epoch)


