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
from classification.models.pytorch.dpn import DPN92
from classification.models.pytorch.mobilenetv2 import MobileNetV2
from classification.models.pytorch.densenet import DenseNet201
from torch.autograd import Variable
from classification.skin.pytorch.data_reader_isic import ISIC2017Dataset

class SkinLeisonClassfication(object):
    def __init__(self,logs):
        self.device_ids=[0]
        self.batch_size_train_per_gpu = 50
        self.batch_size_test_per_gpu=2
        self.epochs = 200
        self.num_classes = 2
        self.log_dir=logs

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.use_cuda = torch.cuda.is_available()
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.model_name_str = None

    def load_data(self):
        # Data
        print('==> Preparing data..')

        trainset = ISIC2017Dataset(task=1, mode='train')
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train_per_gpu, shuffle=True,
                                                  num_workers=2)

        testset = ISIC2017Dataset(task=1, mode='test')
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test_per_gpu, shuffle=False, num_workers=2)

        train_count = len(self.trainloader) * self.batch_size_train_per_gpu
        test_count = len(self.testloader) * self.batch_size_test_per_gpu
        print('==> Total examples, train: {}, test:{}'.format(train_count, test_count))

    def load_model(self, model):
        self.learning_rate=model.learning_rate
        model_name = model.model_name
        model_name_str = model_name.class_name()
        print('\n==> using model {}'.format(model_name_str))
        self.model_name_str="{}_{}".format(model_name_str,model.model_log_name)

        # Model
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert (os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!')
            checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(self.model_name_str ))
            net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        except Exception as e:
            net = model_name()
            print('==> Building model..')

        if self.use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        self.net=net
        self.criterion = nn.CrossEntropyLoss()

        if model.optimizer=="adam":
            self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate, eps=1e-8)

    # Training
    def train(self, epoch):
        print('\n Training Epoch:{} '.format(epoch))
        net = self.net
        optimizer=self.optimizer
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            step = epoch * len(self.trainloader) + batch_idx
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)

            batch_loss=train_loss / (batch_idx + 1)
            if batch_idx%2==0:
                self.writer.add_scalar('step loss',batch_loss,step)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_loss, 100.*correct/total, correct, total))


    def save_model(self, acc, epoch):
        print('\n Saving new model with accuracy {}'.format(acc))
        state = {
            'net': self.net.module if self.use_cuda else self.net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(self.model_name_str ))

    def test(self, epoch):
        writer=self.writer
        net=self.net
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        target_all=[]
        predicted_all=[]
        print("\ntesting with previous accuracy {}".format(self.best_acc))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            predicted_batch=predicted.eq(targets.data).cpu()
            predicted_reshaped=predicted_batch.numpy().reshape(-1)
            predicted_all=np.concatenate((predicted_all,predicted_reshaped),axis=0)

            targets_reshaped = targets.data.cpu().numpy().reshape(-1)
            target_all = np.concatenate((target_all, targets_reshaped), axis=0)
            total += targets.size(0)
            correct += predicted_batch.sum()

            progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        writer.add_scalar('test accuracy', acc, epoch)
        if acc > self.best_acc:
            pass
        self.save_model(acc, epoch)
        self.best_acc = acc

        # accuracy = metrics.accuracy_score(target_all, predicted_all)
        #
        # print('\n Accuracy: {}'.format(accuracy))

        """
        total sum of confusion matrix value is same as total number items in test set.
        """
        cm = metrics.confusion_matrix(target_all, predicted_all)
        print("Confsusion metrics: {}".format(cm))

        auc = metrics.roc_auc_score(target_all, predicted_all)
        print("Auc {}".format(auc))
        writer.add_scalar('test auc', auc, epoch)

        f1_score = metrics.f1_score(target_all, predicted_all)

        print("F1 Score: {}".format(f1_score))
        writer.add_scalar('F1 Score', f1_score, epoch)


        # average_precision = metrics.average_precision_score(y_true, y_pred)
        #
        # print(average_precision)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        # print(TP)
        # print(TN)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        sensitivity=np.mean(TPR)
        print("Senstivity: {} ".format(sensitivity))
        writer.add_scalar('Sensitivity',sensitivity,epoch)
        #Specificity or true negative rate
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

