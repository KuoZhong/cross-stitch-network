'''
utils:
1.accuracy of every class
2.tsne for features
3.gradient
'''

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import visdom
from PIL import Image


plt.switch_backend('agg')

source_vis = visdom.Visdom( port=8098)
source_win_line_accuracy = None
target_vis = visdom.Visdom(port=8098)
target_win_line_accuracy = None
avg_win_line_accuracy = None
name = 'dis_'
viz = visdom.Visdom(port=8098)


def accuracy_per_class(source_dataloader, target_dataloader,  model, device=None ,num_class=31, epoch=None):
    model.train(False)
    if epoch is None or epoch >= min([len(source_dataloader, target_dataloader)]):
        epoch = min([len(source_dataloader), len(target_dataloader)])
    source_true_preds = [0.0 for i in range(num_class)]
    source_label_num = [0.0 for i in range(num_class)]
    target_true_preds = [0.0 for i in range(num_class)]
    target_label_num = [0.0 for i in range(num_class)]
    source_iter_data = iter(source_dataloader)
    target_iter_data = iter(target_dataloader)

    for i in range(epoch):
        imgs, labels = source_iter_data.next()
        if not device is None:
            imgs = imgs.to(device)
            labels = labels.to(device)
        clabels, dlabels = model(imgs)
        for j in range(num_class):
            source_true_preds[j] += sum(dlabels[labels==j]>0.5)
            source_label_num[j] += sum(labels==j)

        imgs, labels = target_iter_data.next()
        if not device is None:
            imgs = imgs.to(device)
            labels = labels.to(device)
        clabels, dlabels = model(imgs)
        for j in range(num_class):
            target_true_preds[j] += sum(dlabels[labels==j]<=0.5)
            target_label_num[j] += sum(labels == j)

    avg_acc = (float(sum(torch.Tensor(source_true_preds))+sum(torch.Tensor(target_true_preds)))/float(sum(torch.Tensor(source_label_num))+sum(torch.Tensor(target_label_num))))
    source_true_preds = [ float(source_true_preds[i])/float(source_label_num[i]) for i in range(num_class)]
    target_true_preds = [ float(target_true_preds[i])/float(target_label_num[i]) for i in range(num_class)]
    return source_true_preds,target_true_preds,avg_acc


def accuracy_per_class_softmax(source_dataloader, target_dataloader,  model, device=None ,num_class=31, epoch=None):
    model.train(False)
    if epoch is None or epoch >= min([len(source_dataloader, target_dataloader)]):
        epoch = min([len(source_dataloader), len(target_dataloader)])
    source_true_preds = [0.0 for i in range(num_class)]
    source_label_num = [0.0 for i in range(num_class)]
    target_true_preds = [0.0 for i in range(num_class)]
    target_label_num = [0.0 for i in range(num_class)]
    source_iter_data = iter(source_dataloader)
    target_iter_data = iter(target_dataloader)

    for i in range(epoch):
        imgs, labels = source_iter_data.next()
        if not device is None:
            imgs = imgs.to(device)
            labels = labels.to(device)
        clabels, dlabels = model(imgs)
        for j in range(num_class):
            if not dlabels[labels==j].size(0) == 0:
                source_true_preds[j] += (dlabels[labels==j].max(1)[1]==1).sum().cpu().item()
                source_label_num[j] += sum(labels==j)
        imgs, labels = target_iter_data.next()
        if not device is None:
            imgs = imgs.to(device)
            labels = labels.to(device)
        clabels, dlabels = model(imgs)
        for j in range(num_class):
            if not dlabels[labels==j].size(0) == 0:
                target_true_preds[j] += (dlabels[labels==j].max(1)[1]==0).sum().cpu().item()
                target_label_num[j] += sum(labels == j)

    avg_acc = (float(sum(torch.Tensor(source_true_preds))+sum(torch.Tensor(target_true_preds)))/float(sum(torch.Tensor(source_label_num))+sum(torch.Tensor(target_label_num))))
    source_true_preds = [ float(source_true_preds[i])/float(source_label_num[i]) for i in range(num_class)]
    target_true_preds = [ float(target_true_preds[i])/float(target_label_num[i]) for i in range(num_class)]
    return source_true_preds,target_true_preds,avg_acc

def visdom_accuracy_per_class(source_true_preds, target_true_preds, avg_acc, epoch, num_class=31):
    global source_win_line_accuracy
    global target_win_line_accuracy
    global avg_win_line_accuracy
    print(epoch)
    print(source_true_preds)
    print(target_true_preds)
    print(avg_acc)

    if target_win_line_accuracy is None:
        i = 0
        source_win_line_accuracy = source_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([source_true_preds[i]]), \
                                                   win='acc_per_cls', name=name + str(i), update=None, env='source')
        target_win_line_accuracy = target_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([target_true_preds[i]]), \
                                                   win='acc_per_cls', name=name + str(i), update=None, env='target')
        avg_win_line_accuracy = target_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([avg_acc]), \
                                                   win='avg_acc', name='avg_acc', update=None,
                                                   env='avg_acc')
        for i in range(1,num_class):
            source_win_line_accuracy = source_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([source_true_preds[i]]),\
                                                       win=source_win_line_accuracy, name=name+str(i), update='append', env='source')
            target_win_line_accuracy = target_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([target_true_preds[i]]), \
                                                       win=target_win_line_accuracy, name=name + str(i), update='append', env='target')

    else:
        avg_win_line_accuracy = target_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([avg_acc]), \
                                                   win=avg_win_line_accuracy, name='avg_acc', update='append',
                                                   env='avg_acc')
        for i in range(num_class):
            source_win_line_accuracy = source_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([source_true_preds[i]]), \
                                                       win=source_win_line_accuracy, name=name + str(i), update='append', env='source')
            target_win_line_accuracy = target_vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([target_true_preds[i]]), \
                                                       win=target_win_line_accuracy, name=name + str(i), update='append', env='target')






def feat_tsne_per_cls(model, source_dataloader, target_dataloader, device, this_epoch, n_components=2,epoch=None):
    feature_extractor = model
    feature_extractor.train(False)
    if epoch is None or epoch >= min([len(source_dataloader, target_dataloader)]):
        epoch = min([len(source_dataloader), len(target_dataloader)])
    source_iter_data = iter(source_dataloader)
    target_iter_data = iter(target_dataloader)
    iter_datas = [source_iter_data, target_iter_data]
    feats = None
    dlabels = None

    for i in range(epoch):
        for k in range(len(iter_datas)):
            imgs, dlabels_tmp = iter_datas[k].next()
            if not device is None:
                imgs = imgs.to(device)
                dlabels_tmp = dlabels_tmp.to(device)
            feat_tmp = feature_extractor(imgs)
            if feats is None:
                feats = feat_tmp.detach()
                dlabels = dlabels_tmp
            else:
                feats = torch.cat([feats, feat_tmp.detach()])
                dlabels = torch.cat([dlabels, dlabels_tmp])
    feats = feats.detach().cpu().numpy()
    dlabels = dlabels.cpu().float() + 1
    my_feats_tsne = TSNE(n_components=n_components, learning_rate=100).fit_transform(feats)
    assert my_feats_tsne.shape[1]==n_components
    vis = visdom.Visdom(port=8098)
    print('source instance number:  %d'%(sum(dlabels)-len(dlabels)))
    print('target instance number:  %d'%(2 * len(dlabels) - sum(dlabels)))
    print('epoch_'+ str(this_epoch))
    vis.scatter(torch.from_numpy(my_feats_tsne),torch.Tensor(dlabels), win='epoch_'+ str(this_epoch), \
                opts=dict(markersize=6, title='epoch_'+ str(this_epoch)))

def feat_tsne(model, source_dataloader, target_dataloader, device, this_epoch, n_components=2,epoch=None):
    feature_extractor = model
    feature_extractor.train(False)
    if epoch is None or epoch >= min([len(source_dataloader, target_dataloader)]):
        epoch = min([len(source_dataloader), len(target_dataloader)])
    source_iter_data = iter(source_dataloader)
    target_iter_data = iter(target_dataloader)
    iter_datas = [source_iter_data, target_iter_data]
    feats = None
    dlabels = None

    for i in range(epoch):
        for k in range(len(iter_datas)):
            imgs, _ = iter_datas[k].next()
            if not device is None:
                imgs = imgs.to(device)
                #labels = labels.to(device)
            feat_tmp = feature_extractor(imgs)
            if feats is None:
                feats = feat_tmp.detach()
                dlabels = [2 for i in range(imgs.size(0))]
            else:
                feats = torch.cat([feats, feat_tmp.detach()])
                if k == 0:
                    dlabels += [2 for i in range(imgs.size(0))]
                else:
                    dlabels += [1 for i in range(imgs.size(0))]

    feats = feats.detach().cpu().numpy()
    my_feats_tsne = TSNE(n_components=n_components, learning_rate=100).fit_transform(feats)
    assert my_feats_tsne.shape[1]==n_components
    vis = visdom.Visdom(port=8098)
    print('source instance number:  %d'%(sum(dlabels)-len(dlabels)))
    print('target instance number:  %d'%(2 * len(dlabels) - sum(dlabels)))
    print('epoch_'+ str(this_epoch))
    vis.scatter(torch.from_numpy(my_feats_tsne),torch.Tensor(dlabels), win='epoch_'+ str(this_epoch), \
                opts=dict(markersize=6, title='epoch_'+ str(this_epoch)))


'''
    for i in range(feats_tsne.shape[0]):
        if dlabels[i]==1:
            plt.plot(feats_tsne[i, 0], feats_tsne[i, 1], 'r', label="point", markersize=12)
        else:
            plt.plot(feats_tsne[i, 0], feats_tsne[i, 1], 'b', label="point", markersize=12)
    plt.savefig(os.path.join(root_path,str(this_epoch)+'_feat_tsne.png'), dpi=100)

    vis = visdom.Visdom( port=8098)
    img = np.array(Image.open(os.path.join(root_path,str(this_epoch)+'_feat_tsne.png')))
    vis.image(torch.Tensor(img), env='epoch_'+ str(this_epoch))
'''
#after backward
def get_weight_norm(m):
    param_num = 0.0
    grad_norm = 0.0
    for name, child_mod in m.named_children():
        param_num_tmp, grad_norm_tmp = get_weight_norm(child_mod)
        param_num += param_num_tmp
        grad_norm += grad_norm_tmp
    if not param_num == 0.0:
        return param_num,grad_norm
    param_num = 1.0
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        size = m.weight.grad.size()
        for i in range(len(size)):
            param_num *= size[i]
        grad_norm = float((m.weight.grad.norm() ** 2).cpu().item())
        return param_num, grad_norm
    else:
        return 0,0

# since computation graph will be freed, backward here should keep graph and this test should be used before being optimized
def gradient_norm(loss, model, epoch, name='grad_norm'):
    model.zero_grad()
    loss.backward(retain_graph=True)
    param_num, grad_norm = get_weight_norm(model)
    assert not param_num == 0
    print('grad',epoch,'||',param_num,'|||',grad_norm/param_num)

    if epoch == 1:
        viz.line(X=torch.Tensor([epoch]), Y=torch.Tensor([grad_norm/param_num]), win='grad_norm', name=name, update=None)
    else:
        viz.line(X=torch.Tensor([epoch]), Y=torch.Tensor([grad_norm/param_num]), win='grad_norm', name=name, update='append')




