import torch
import model
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import os
import math
import data_list
from torchvision import transforms
import torch.utils.data
import utils_visulization
import cross_stitch_network


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
seed = 0
batch_size = 64
num_workers = 4
epochs = 1000
lr = 1.0e-4
momentum = 0.9
no_cuda =False
seed = 0
pretrain = True
log_interval = 10
l2_decay = 5e-4
module_path = {
    'source': '/home/neon/experiment/cross-stitch-network/model/webcam/',
    'target': '/home/neon/experiment/cross-stitch-network/model/amazon/',
    'base': '/home/neon/experiment/cross-stitch-network/model/cross/'
}  # only set here
dataset_link_file_path = {
    'source': '/home/neon/dataset/office/webcam/',
    'target': '/home/neon/dataset/office/amazon/'
}  # only set here
loss = torch.nn.CrossEntropyLoss()
device = None
class_num = 31
save_best_model = True
drop_last = True
model_pretrained = {
    'source': True,
    'target': True
}
interval = 50
max_iteration = 20000


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

dataset_train = {tmp: data_list.ImageList(open(os.path.join(dataset_link_file_path[tmp], 'train.txt')).readlines(),
                                          transform=transform_train) for tmp in dataset_link_file_path}
dataset_test = {tmp: data_list.ImageList(open(os.path.join(dataset_link_file_path[tmp], 'test.txt')).readlines(),
                                         transform=transform_test) for tmp in dataset_link_file_path}
dataloader_train = {tmp: torch.utils.data.DataLoader(dataset_train[tmp], batch_size=batch_size, shuffle=True,
                                                     num_workers=num_workers, drop_last=drop_last) for tmp in dataset_train}
dataloader_test = {tmp: torch.utils.data.DataLoader(dataset_test[tmp], batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers) for tmp in dataset_test}
len_dataset_train = {tmp: len(dataset_train[tmp]) for tmp in dataset_train}
len_dataset_test = {tmp: len(dataset_test[tmp]) for tmp in dataset_test}


class_num = 31

#model
m_models = {tmp:model.network_dict['AlexNetFc'](True) for tmp in dataset_train}
m_models = {tmp:nn.Sequential(m_models[tmp], nn.Linear(m_models[tmp].out_features, class_num)) for tmp in dataset_train}
# load pretrained parameters
for tmp in m_models:
    m_models[tmp].load_state_dict(torch.load(os.path.join(module_path[tmp], 'best_model.pth')))

base_cross_stitch_net = cross_stitch_network.CrossStitchNetwork(m_models['source'][0], m_models['target'][0])
classifiers = {tmp: m_models[tmp][1] for tmp in m_models}
if cuda:
    m_models = {tmp: m_models[tmp].cuda() for tmp in m_models}
    base_cross_stitch_net = base_cross_stitch_net.cuda()
    classifiers = {tmp: m_models[tmp][1].cuda() for tmp in m_models}

paramter_list = [
    {'params': m_models['source'][0].parameters(), 'lr':1e-5},
    {'params': m_models['target'][0].parameters(), 'lr': 1e-5},
    {'params': m_models['source'][1].parameters(), 'lr': 10e-5},
    {'params': m_models['target'][1].parameters(), 'lr': 10e-5},
    {'params': base_cross_stitch_net.cross_stitch_units.parameters(), 'lr': 1.0e-2}
]
# for i in range(len(base_cross_stitch_net.cross_stitch_units)):
#     paramter_list.append({'params':base_cross_stitch_net.cross_stitch_units[i].parameters()})
optimizer = optim.Adam(paramter_list, lr=lr)


def test(base_cross_stitch_net, m_classifier, data_loader, is_source=True):
    base_cross_stitch_net.train(False)
    m_classifier.train(False)
    test_loss = 0
    correct = 0
    len_data_loader = 0

    for data, target in data_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if is_source:
            s_output,_= base_cross_stitch_net(data)
        else:
            _, s_output = base_cross_stitch_net(data)
        s_output = m_classifier(s_output)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        len_data_loader += data.size(0)

    test_loss /= len_data_loader
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        'this', test_loss, correct, len_data_loader,
        100.0 * float(correct) / len_data_loader))
    return correct

# train model
def train(base_cross_stitch_net, classifiers, dataloader_train, dataloader_test, max_iteration, interval):
    iter_train = {tmp: iter(dataloader_train[tmp]) for tmp in dataloader_train}
    len_dataloader_train = {tmp: len(dataloader_train[tmp]) for tmp in dataloader_train}

    for m_iteration in range(max_iteration):
        if m_iteration % interval == 0:
            # torch.save(base_cross_stitch_net.state_dict(), os.path.join(module_path['base'], 'last_model.pth'))
            # for tmp in classifiers:
            #     torch.save(classifiers[tmp].state_dict(), os.path.join(module_path[tmp], tmp + 'last_model.pth'))
            correct = {}
            for tmp in dataloader_test:
                print(tmp)
                is_source = True
                if tmp == 'target':
                    is_source = False
                correct[tmp] = test(base_cross_stitch_net, classifiers[tmp], dataloader_test[tmp], is_source)
                utils_visulization.classification_accuracy_plot(torch.Tensor([m_iteration]),
                                                                torch.Tensor([100.0 * float(
                                                                    correct[tmp]) / len_dataset_test[tmp]]), \
                                                                name='accuracy_'+tmp)

        base_cross_stitch_net.train(True)
        {tmp: classifiers[tmp].train(True) for tmp in classifiers}

        for tmp in dataloader_train:
            if m_iteration % len_dataloader_train[tmp] == 0:
                iter_train[tmp] = iter(dataloader_train[tmp])
        # print(m_iteration)
        source_imgs, source_labels = iter_train['source'].next()
        target_imgs, target_labels = iter_train['target'].next()
        imgs = torch.cat([source_imgs, target_imgs], dim=0)
        if cuda:
            imgs = imgs.cuda()
            source_labels,target_labels = source_labels.cuda(), target_labels.cuda()
        feats1, feats2 = base_cross_stitch_net(imgs)
        feats1 = torch.narrow(feats1, 0, 0, feats1.size(0)//2)
        feats2 = torch.narrow(feats2, 0, feats2.size(0)//2, feats2.size(0)//2)
        feats1, feats2 = classifiers['source'](feats1), classifiers['target'](feats2)
        # print(feats1.shape, feats2.shape)
        # print(m_iteration)
        m_loss = (loss(feats1, source_labels) + loss(feats2, target_labels))/2.0
        optimizer.zero_grad()
        m_loss.backward()
        # optimizer.step()
        utils_visulization.loss_plot(torch.Tensor([m_iteration]), torch.Tensor([m_loss]), name='train_loss')


if __name__ == '__main__':
    train(base_cross_stitch_net,classifiers,dataloader_train,dataloader_test,max_iteration, interval)