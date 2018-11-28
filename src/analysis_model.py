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
import numpy as np
import matplotlib.pyplot as plt

import numpy
from numpy import *
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class_names = ['back_pack',
 'bike',
 'bike_helmet',
 'bookcase',
 'bottle',
 'calculator',
 'desk_chair',
 'desk_lamp',
 'desktop_computer',
 'file_cabinet',
 'headphones',
 'keyboard',
 'laptop_computer',
 'letter_tray',
 'mobile_phone',
 'monitor',
 'mouse',
 'mug',
 'paper_notebook',
 'pen',
 'phone',
 'printer',
 'projector',
 'punchers',
 'ring_binder',
 'ruler',
 'scissors',
 'speaker',
 'stapler',
 'tape_dispenser',
 'trash_can']

# Training settings
seed = 0
batch_size = int(os.environ['BATCH_SIZE'])
num_workers = 4
no_cuda =False
seed = 0
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
class_num = 31
drop_last = False
interval = int(os.environ['LOG_INTERVAL'])
cuda = not no_cuda and torch.cuda.is_available()
other_file_path = "/opt/ml/disk/output/MisclassifiedSample/"
ellipsis = 1.0e-6

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

transform_test = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor()]
)


dataset_test = {tmp: data_list.ImageListWithFilePath(open(os.path.join(dataset_link_file_path[tmp], 'test.txt')).readlines(),
                                         transform=transform_test) for tmp in dataset_link_file_path}

dataloader_test = {tmp: torch.utils.data.DataLoader(dataset_test[tmp], batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers) for tmp in dataset_test}

len_dataset_test = {tmp: len(dataset_test[tmp]) for tmp in dataset_test}


#model
m_models = {tmp:model.network_dict['AlexNetFc'](True) for tmp in dataset_test}
m_models = {tmp:nn.Sequential(m_models[tmp], nn.Linear(m_models[tmp].out_features, class_num)) for tmp in dataset_test}

base_progress_net = progress_net.ProgressNetwork(m_models['source'][0], m_models['target'][0])
classifiers = {tmp: m_models[tmp][1] for tmp in m_models}

# need reset path
base_progress_net.load_state_dict(torch.load(os.path.join(cloud_module_path['base'], 'best_model.pth')))
for tmp in classifiers:
    print(classifiers[tmp])
    classifiers[tmp].load_state_dict(torch.load(os.path.join(cloud_module_path['base'], 'imgnet_scratch_progress_' + tmp + 'best_model.pth')))

if cuda:
    m_models = {tmp: m_models[tmp].cuda() for tmp in m_models}
    base_progress_net = base_progress_net.cuda()
    classifiers = {tmp: m_models[tmp][1].cuda() for tmp in m_models}



def find_misclassified_examples(base_progrese_net, m_classifier, data_loader):
    base_progress_net.train(False)
    m_classifier.train(False)
    results = {i:[] for i in range(class_num)}
    num_error = 0
    for imgs, labels, paths in data_loader:
        if cuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        _, feats = base_progrese_net(imgs)
        preds = m_classifier(feats)
        preds = preds.data.max(1)[1].view(preds.size(0))
        # print(preds)
        # print(labels)
        # print(paths)
        for i in range(preds.size(0)):
            if preds[i].item() != labels[i].item():
                print(preds[i].item())
                print(labels[i].item())
                results[labels[i].item()].append(paths[i])
                num_error += 1
    print(results)
    for i in range(class_num):
        if len(results[i]) != 0:
            f = open(os.path.join(other_file_path, str(i) + '.txt'), 'w')
            tmp = [path + '\n' for path in results[i]]
            f.writelines(tmp)
            f.close()

    return results

def find_model_difference_pca(base_progrese_net, data_loader):
    base_progress_net.train(False)
    feat_source_class = [None for i in range(class_num)]
    feat_tareget_class = [None for i in range(class_num)]
    with torch.no_grad():
        for imgs, labels, _ in data_loader:
            imgs = imgs.cuda()
            feat_source, feat_target = base_progress_net(imgs)
            for i in range(feat_source.size(0)):
                if feat_source_class[labels[i].item()] is None:
                    feat_source_class[labels[i].item()] = feat_source[i, :].view(1, -1).detach()
                    feat_tareget_class[labels[i].item()] = feat_target[i, :].view(1, -1).detach()
                else:
                    feat_source_class[labels[i].item()] = torch.cat((feat_source_class[labels[i].item()], feat_source[i, :].view(1, -1).detach()), dim=0)
                    feat_tareget_class[labels[i].item()] = torch.cat((feat_tareget_class[labels[i].item()], feat_target[i, :].view(1, -1).detach()), dim=0)
        big_feat = torch.cat(feat_source_class + feat_tareget_class, dim = 0)
        # pca_process
        # centralize
        big_mean = torch.mean(big_feat, 0, True).detach()
        big_feat = big_feat - big_mean
        feat_source_class = [feat_source_class[i] - big_mean for i in range(class_num)]
        feat_tareget_class = [feat_tareget_class[i] - big_mean for i in range(class_num)]
        #normalize
        big_std = torch.std(big_feat, 0, True).detach()
        big_feat = big_feat / big_std
        feat_source_class = [feat_source_class[i]/big_std for i in range(class_num)]
        feat_tareget_class = [feat_tareget_class[i]/big_std for i in range(class_num)]

        big_feat_square = torch.mm(torch.t(big_feat), big_feat) / big_feat.size(0)
        print(big_feat_square.max())
        # ellipsis_diag_matrix = (torch.eye(big_feat_square.size(0))*ellipsis).cuda()
        #
        # big_feat_square = big_feat_square + ellipsis_diag_matrix
        # #SVD
        # U, S, V = torch.svd(big_feat_square)
        # U = U[:,:2].detach()
        big_feat_square_numpy = big_feat_square.cpu().numpy()
        ellipsis_diag_matrix = ellipsis * np.eye(big_feat_square.size(0))
        big_feat_square_numpy = ellipsis_diag_matrix + big_feat_square_numpy
        U, _, _ = np.linalg.svd(big_feat_square_numpy)
        U = U[:, :2]
        U = torch.from_numpy(U)
        print(U.shape)

        #projection
        feat_source_class = [torch.mm(feat_source_class[i], U) for i in range(class_num)]
        feat_tareget_class = [torch.mm(feat_tareget_class[i], U) for i in range(class_num)]

        #consider distance between classes
        avg_feat_source_class = [torch.mean(feat_source_class[i], 0, True) for i in range(class_num)]
        avg_feat_target_class = [torch.mean(feat_tareget_class[i], 0, True) for i in range(class_num)]

        fig = plt.figure()
        tmp_ax = fig.add_subplot(6,6,1)
        tmp_ax.set_title('distance between classes')
        for i in range(class_num):
            tmp_target_scatter = avg_feat_target_class[i].cpu().numpy()
            tmp_ax.scatter(tmp_target_scatter[0, 0], tmp_target_scatter[0, 1], c='r', marker='x',  alpha=.5, label='target')
            tmp_source_scatter = avg_feat_source_class[i].cpu().numpy()
            tmp_ax.scatter(tmp_source_scatter[0, 0], tmp_source_scatter[0, 1], c='b', marker='x',  alpha=.5, label='source')
            tmp_ax.plot([tmp_source_scatter[0, 0], tmp_target_scatter[0, 0]], [tmp_source_scatter[0, 1], tmp_target_scatter[0, 1]], alpha= .5)

        for i in range(class_num):
            tmp_ax = fig.add_subplot(6, 6, i+2)
            tmp_ax.set_title('in the class ' + str(i) + 'th: ' + class_names[i])
            tmp_target_scatter = feat_tareget_class[i].cpu().numpy()
            tmp_ax.plot(tmp_target_scatter[:,0], tmp_target_scatter[:,1], c='r', marker='x', alpha=.5, label='target')
            tmp_source_scatter = feat_source_class[i].cpu().numpy()
            tmp_ax.plot(tmp_source_scatter[:,0], tmp_source_scatter[:,1], c='b', marker='x', alpha=.5, label='source')

            tmp_target_scatter = avg_feat_target_class[i].cpu().numpy()
            tmp_ax.scatter(tmp_target_scatter[0, 0], tmp_target_scatter[0, 1], c='r', marker='o', markersize=20, alpha=1.0,
                           label='target_avg')
            tmp_source_scatter = avg_feat_source_class[i].cpu().numpy()
            tmp_ax.scatter(tmp_source_scatter[0, 0], tmp_source_scatter[0, 1], c='b', marker='o', markersize=20, alpha=1.0,
                           label='source_avg')


        fig.savefig(os.path.join(other_file_path, 'diff.png'))


def tsne(input):
    input = input.cpu().numpy()
    output = TSNE(n_components=2).fit_transform(input)
    return output


def find_model_difference_pca(base_progrese_net, data_loader):
    base_progress_net.train(False)
    feat_source_class = [None for i in range(class_num)]
    feat_tareget_class = [None for i in range(class_num)]
    labels = None
    big_feat = None
    with torch.no_grad():
        for imgs, labels, _ in data_loader:
            imgs = imgs.cuda()
            feat_source, feat_target = base_progress_net(imgs)
            if big_feat is None:
                big_feat = torch.cat([feat_source, feat_target], dim=0)
                labels = torch.cat([labels+class_num, labels], dim=0)
            else:
                big_feat = torch.cat([big_feat, feat_source, feat_target], dim=0)
                labels = torch.cat([labels, feat_source, feat_target], dim=0)
    x_tsne = tsne(big_feat.cpu().numpy())
    labels = labels.numpy()
    




if __name__ == '__main__':
    find_model_difference(base_progress_net, dataloader_test['target'])
