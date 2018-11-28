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
import progress_net

# Training settings
seed = 0
batch_size = int(os.environ['BATCH_SIZE'])
num_workers = 4
#epochs = 1000
lr = 1.0e-4
momentum = 0.9
no_cuda =False
seed = 0
l2_decay = 5e-4
save_module_path = {
    'source': '/opt/ml/model/',
    'target': '/opt/ml/model/',
    'base': '/opt/ml/model/'
}  # only set here
dataset_link_file_path = {
    'source': os.path.join('./dataset_list/', os.environ['SOURCE']),
    'target': os.path.join('./dataset_list/', os.environ['TARGET'])
}  # only set here
cloud_module_path = {
    'source': os.path.join('/opt/ml/disk/model/alexnet', os.environ['SOURCE']),
    'target': os.path.join('/opt/ml/disk/model/alexnet', os.environ['TARGET']),
    'base': '/opt/ml/disk/model/progress_back'
}
loss = torch.nn.CrossEntropyLoss()
device = None
class_num = 31
save_best_model = True
drop_last = True
interval = int(os.environ['LOG_INTERVAL'])
max_iteration = int(os.environ['EPOCHS'])
load_pretrained_model = bool(int(os.environ['PRETRAINED']))
save_model = True
cross_unit_lr = float(os.environ['CROSS_UNIT_LR'])
back_lr = float(os.environ['BACK_LR'])
from_scratch = bool(int(os.environ['FROM_SCRATCH']))

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
if from_scratch:
    torch.nn.init.kaiming_normal_(m_models['target'][1].weight.data)
    m_models['target'][1].bias.data.fill_(0)
else:
    m_models['target'].load_state_dict(torch.load(os.path.join(cloud_module_path['target'], 'best_model.pth')))

base_progress_net = progress_net.ProgressNetwork(m_models['source'][0], m_models['target'][0])
classifiers = {tmp: m_models[tmp][1] for tmp in m_models}

# need reset path
if load_pretrained_model:
    base_progress_net.load_state_dict(torch.load(os.path.join(cloud_module_path['base'], 'last_model.pth')))
    for tmp in classifiers:
        print(classifiers[tmp])
        classifiers[tmp].load_state_dict(torch.load(os.path.join(cloud_module_path['base'], 'imgnet_scratch_progress_' + tmp + 'last_model.pth')))

if cuda:
    m_models = {tmp: m_models[tmp].cuda() for tmp in m_models}
    base_progress_net = base_progress_net.cuda()
    classifiers = {tmp: m_models[tmp][1].cuda() for tmp in m_models}

paramter_list = [
    # {'params': m_models['source'][0].parameters(), 'lr':1e-5},
    {'params': m_models['target'][0].parameters(), 'lr': back_lr},
    # {'params': m_models['source'][1].parameters(), 'lr': 10.0 * back_lr},
    {'params': m_models['target'][1].parameters(), 'lr': 10.0 * back_lr},
    {'params': base_progress_net.progress_units.parameters(), 'lr': cross_unit_lr}
]
# for i in range(len(base_cross_stitch_net.cross_stitch_units)):
#     paramter_list.append({'params':base_cross_stitch_net.cross_stitch_units[i].parameters()})
# optimizer = optim.Adam(paramter_list, lr=lr)
optimizer = optim.Adam(paramter_list, lr=lr)


def test(base_progrese_net, m_classifier, data_loader, num_iter, is_source=True):
    base_progress_net.train(False)
    m_classifier.train(False)
    test_loss = 0
    correct = 0
    len_data_loader = 0

    for data, target in data_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if is_source:
            s_output,_= base_progress_net(data)
        else:
            _, s_output = base_progress_net(data)
        s_output = m_classifier(s_output)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        len_data_loader += data.size(0)

    test_loss /= len_data_loader
    if is_source:
        data_name = os.environ['SOURCE']
    else:
        data_name = os.environ['TARGET']
    print('\n{} iteration {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(num_iter, data_name
        , test_loss, correct, len_data_loader,
        100.0 * float(correct) / len_data_loader))
    return correct

# train model
def train(base_progress_net, classifiers, dataloader_train, dataloader_test, max_iteration, interval):
    iter_train = {tmp: iter(dataloader_train[tmp]) for tmp in dataloader_train}
    len_dataloader_train = {tmp: len(dataloader_train[tmp]) for tmp in dataloader_train}
    max_correct = 0.0

    for m_iteration in range(max_iteration):
        if m_iteration % interval == 0:
            if save_model:
                torch.save(base_progress_net.state_dict(), os.path.join(save_module_path['base'], 'last_model.pth'))
                for tmp in classifiers:
                    torch.save(classifiers[tmp].state_dict(), os.path.join(save_module_path[tmp], 'imgnet_scratch_progress_' + tmp + 'last_model.pth'))
            # correct = {}
            # for tmp in dataloader_test:
            #     # print(tmp)
            #     is_source = True
            #     if tmp == 'target':
            #         is_source = False
            #     correct[tmp] = test(base_progress_net, classifiers[tmp], dataloader_test[tmp], is_source)
            #     utils_visulization.classification_accuracy_plot(torch.Tensor([m_iteration]),
            #                                                     torch.Tensor([100.0 * float(
            #                                                         correct[tmp]) / len_dataset_test[tmp]]), \
            #                                                     name='accuracy_'+tmp)
            correct = test(base_progress_net, classifiers['target'], dataloader_test['target'], m_iteration, False)
            if correct > max_correct:
                max_correct = correct
            print('\n{} iteration {} set: max Accuracy: {}/{} ({:.2f}%)\n'.format(m_iteration, os.environ['TARGET'],
                                                                                  max_correct,
                                                                                  len_dataset_test['target'],
                                                                                  100.0 * float(max_correct) / len_dataset_test['target']))

        # base_progress_net.train(True)
        base_progress_net.target_architecture.train(True)
        base_progress_net.progress_units.train(True)
        base_progress_net.source_architecture.train(False)
        classifiers['target'].train(True)
        # {tmp: classifiers[tmp].train(True) for tmp in classifiers}

        for tmp in dataloader_train:
            if m_iteration % len_dataloader_train[tmp] == 0:
                iter_train[tmp] = iter(dataloader_train[tmp])
        # print(m_iteration)
        # source_imgs, source_labels = iter_train['source'].next()
        target_imgs, target_labels = iter_train['target'].next()
        imgs = target_imgs
        # imgs = torch.cat([source_imgs, target_imgs], dim=0)
        if cuda:
            imgs = imgs.cuda()
            target_labels = target_labels.cuda()
        feats1, feats2 = base_progress_net(imgs)
        # feats1 = torch.narrow(feats1, 0, 0, feats1.size(0)//2)
        # feats2 = torch.narrow(feats2, 0, feats2.size(0)//2, feats2.size(0)//2)
        feats1, feats2 = classifiers['source'](feats1), classifiers['target'](feats2)
        # print(feats1.shape, feats2.shape)
        # print(m_iteration)
        m_loss = loss(feats2, target_labels)
        optimizer.zero_grad()
        m_loss.backward()
        optimizer.step()
        # utils_visulization.loss_plot(torch.Tensor([m_iteration]), torch.Tensor([m_loss]), name='train_loss')
        print('\n{} iteration {} set: loss: ({:.2f}%)\n'.format(m_iteration, os.environ['TARGET'],
                                                                100.0 * float(max_correct) / len_dataset_test['target']))


if __name__ == '__main__':
    train(base_progress_net,classifiers,dataloader_train,dataloader_test,max_iteration, interval)
