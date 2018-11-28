import torch
import model
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_list
from torchvision import transforms


# Training settings
seed = 0
batch_size = 128
num_workers = 4
epochs = 2000
lr = 0.01
momentum = 0.9
no_cuda =False
seed = 0
pretrain = True
log_interval = 10
l2_decay = 5e-4
module_path = '/opt/ml/model/' # only set here
dataset_link_file_path = './dataset_list/amazon/'# only set here
loss = torch.nn.CrossEntropyLoss()
device = None
class_num = 31
save_best_model = True


cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)


transform_train = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
)
transform_test = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor()]
)

dataset_train = data_list.ImageList(open(os.path.join(dataset_link_file_path, 'train.txt')).readlines(), transform=transform_train)
dataset_test = data_list.ImageList(open(os.path.join(dataset_link_file_path, 'test.txt')).readlines(), transform=transform_test)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
len_dataset_train = len(dataset_train)
len_dataset_test = len(dataset_test)

feat_net = model.network_dict['AlexNetFc'](pretrain).cuda()
cls_layer = torch.nn.Linear(feat_net.out_features, class_num)
torch.nn.init.kaiming_normal_(cls_layer.weight.data)
cls_layer = cls_layer.cuda()

m_model = torch.nn.Sequential(feat_net,cls_layer)
#m_model.load_state_dict(torch.load(os.path.join(module_path, 'best_model.pth')))

optimizer = optim.Adam([
    {'params': feat_net.parameters(), 'lr':1.0e-6},
    {'params': cls_layer.parameters(), 'lr':10.0e-6}
], )


def train_per_epoch(feat_net, cls_layer, m_dataloader, loss, optimizer, no_cuda):
    feat_net.train(True)
    cls_layer.train(True)
    losses = []
    for imgs,labels in m_dataloader:
        # labels = torch.Tensor(labels)
        if not no_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        feats = feat_net(imgs)
        preds = cls_layer(feats)
        m_cross_entropy_loss = loss(preds, labels)
        losses.append(float(m_cross_entropy_loss.item()))
        optimizer.zero_grad()
        m_cross_entropy_loss.backward()
        optimizer.step()
    return sum(losses)/float(len(losses))

def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    len_data_loader = 0

    for data, target in data_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output= model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        len_data_loader += data.size(0)

    test_loss /= len_data_loader
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        'this', test_loss, correct, len_data_loader,
        100.0 * float(correct) / len_data_loader))
    return correct


if __name__ == '__main__':
    m_model = torch.nn.Sequential(feat_net, cls_layer)
    correct = 0
    for epoch in range(1, epochs + 1):
        avg_loss = train_per_epoch(feat_net, cls_layer, dataloader_train, loss, optimizer, no_cuda)
        # utils_visulization.loss_plot(torch.Tensor([epoch]), torch.Tensor([avg_loss]), name='train_loss')
        print('epoch:{}  loss:{: .3f}\n' .format(epoch, avg_loss))

        #train-set
        t_correct_train = test(m_model, dataloader_train)
        # utils_visulization.classification_accuracy_plot(torch.Tensor([epoch]),
        #                                                 torch.Tensor([100.0 * float(t_correct_train) / len_dataset_train]), \
        #                                                 name='accuracy_train')
        print('train_set: epoch:{} acc:{: .2f}%\n'.format(epoch, 100.0 * float(t_correct_train) / len_dataset_train))

        #test-set
        t_correct = test(m_model, dataloader_test)
        # utils_visulization.classification_accuracy_plot(torch.Tensor([epoch]),
        #                                                 torch.Tensor([100.0 * float(t_correct) / len_dataset_test]), \
        #                                                 name='accuracy_test')
        print('test_set: epoch:{} acc:{: .2f}%\n'.format(epoch, 100.0 * float(t_correct) / len_dataset_test))

        if t_correct > correct:
            correct = t_correct
            if save_best_model:
                torch.save(m_model.state_dict(), os.path.join(module_path, 'best_model.pth'))

        print('epoch:{} correct: {}  accuracy{: .2f}%\n'.format(
            epoch, t_correct, 100.0 * float(t_correct) / len_dataset_test))
        print(' max correct: {} max accuracy{: .2f}%\n'.format(
            correct, 100.0 * float(correct) / len_dataset_test))
        torch.save(m_model.state_dict(), os.path.join(module_path, 'last_model.pth'))
