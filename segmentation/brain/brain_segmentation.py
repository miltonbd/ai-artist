from __future__ import print_function

from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from utils.functions import progress_bar

import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from segmentation.brain.data_reader_brats import BratsDataset
from segmentation.models.pytorch.unet_model2 import UNet2 as UNet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import imageio

class Segmentation(object):
    def __init__(self,logs):
        self.device_ids=[0]
        self.batch_size_train_per_gpu = 60
        self.batch_size_test_per_gpu=1
        self.epochs = 200
        self.learning_rate = 0.01
        self.log_dir=logs

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.use_cuda = torch.cuda.is_available()
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.model_name_str = None

    def load_data(self):
        # Data
        print('==> Preparing data..')

        trainset = BratsDataset( mode='train')
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train_per_gpu, shuffle=True,
                                                  num_workers=2)

        testset = BratsDataset(mode='test')
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test_per_gpu, shuffle=False, num_workers=2)

        train_count = len(self.trainloader) * self.batch_size_train_per_gpu
        test_count = len(self.testloader) * self.batch_size_test_per_gpu
        print('==> Total examples, train: {}, test:{}'.format(train_count, test_count))

    def load_model(self):
        model_name_str = "unet2"
        model_log_name='adam1'
        print('\n==> using model {}'.format(model_name_str))
        self.model_name_str="{}_{}".format(model_name_str,model_log_name)
        self.model_path='./checkpoint/{}_ckpt.t7'.format(self.model_name_str )

        # Model
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert (os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!')
            checkpoint = torch.load(self.model_path)
            if not os.path.exists(checkpoint):
                raise Exception('ok')

            model = checkpoint['model']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        # except FileNotFoundError as e:
        #     net = UNet(3, depth=5, merge_mode='concat')
        #     print('==> Building model..')
        except Exception as e:
            print(e)
            print(e.__traceback__)
            model = UNet(1, 1)
            print('==> Building model..')

        if self.use_cuda:
            model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=5)

    # Training
    def train(self, epoch):
        print('\n Training Epoch:{} '.format(epoch))
        model = self.model
        optimizer=self.optimizer
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            step = epoch * len(self.trainloader) + batch_idx
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model.forward(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            self.writer.add_scalar('train loss',train_loss,step)
            total += targets.size(0)
            #correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def save_model(self, acc, epoch):
        print('\n Saving new model with accuracy {}'.format(acc))
        state = {
            'model': self.model.module if self.use_cuda else self.model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint'):
            os.mkdir('checkpoint')
        print(self.model_path)
        torch.save(state, self.model_path)

    def test(self, epoch):
        writer=self.writer
        model=self.model
        model.eval()
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
            outputs = model(inputs)
            output_mask=outputs.cpu()[0,:,:].data.numpy()
            #output_mask = (output_mask > .9)*1.0
            imageio.imwrite("/home/milton/dataset/segmentation/BRATS/BRATS2015/output/out_{}.jpg".format(batch_idx), output_mask)
            #print("loaded")
            # loss = self.criterion(outputs, targets)
            #
            # test_loss += loss.data[0]
            # _, predicted = torch.max(outputs.data, 1)
            # predicted_batch=predicted.eq(targets.data).cpu()
            # predicted_reshaped=predicted_batch.numpy().reshape(-1)
            # predicted_all=np.concatenate((predicted_all,predicted_reshaped),axis=0)
            #
            # targets_reshaped = targets.data.cpu().numpy().reshape(-1)
            # target_all = np.concatenate((target_all, targets_reshaped), axis=0)
            # total += targets.size(0)
            # correct += predicted_batch.sum()
            #
            # progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.save_model(1, epoch)
        return
        exit(1)
        # Save checkpoint.
        acc = 100.*correct/total
        writer.add_scalar('test accuracy', acc, epoch)
        if acc > self.best_acc:
            pass
        self.save_model(acc, epoch)
        self.best_acc = acc

        accuracy = metrics.accuracy_score(target_all, predicted_all)

        print('\n Accuracy: {}'.format(accuracy))

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

if __name__ == '__main__':
    trainer=Segmentation('logs/adam1')
    trainer.load_data()
    trainer.load_model()
    for epoch in range(trainer.start_epoch, trainer.start_epoch + trainer.epochs):
        try:
            trainer.train(epoch)
            trainer.test(epoch)
        except KeyboardInterrupt:
            trainer.test(epoch)
            break;
        # clasifier.load_data()
