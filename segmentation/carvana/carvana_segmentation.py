import torch
from torch import optim,nn
from torch.autograd import Variable
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from segmentation.carvana.data_reader_cardava import CarvanaDataset
from segmentation.models.pytorch.unet_model2 import UNet2

batch_size_train_per_gpu=2
learning_rate=0.001
weight_decay=1e-5

trainset = CarvanaDataset(mode='train')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train_per_gpu, shuffle=True,
                                               num_workers=2)
model=UNet2(3,1)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.BCELoss().cuda()

def train(epoch):
    avg_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs, targets = data.cuda(), target.cuda()
        data, target = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        output = model(data)
        # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]
        num_batches += 1
        # if batch_idx % 109 == 0:
        #     print(loss.data[0])
    avg_loss /= num_batches
    print("loss {}".format(avg_loss))


if __name__ == '__main__':
    for epoch in range(0, 10):
        try:
           train(epoch)

        except KeyboardInterrupt:

            break;
        # clasifier.load_data()








