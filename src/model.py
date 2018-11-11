import torch
import torch.nn as nn
import torchvision.models as models

network_dict = {}


def add_net(cls):
    network_dict[cls.__name__] = cls
    return cls


@add_net
class AlexNetFc(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained)
        self.features = model_alexnet.features
        self.classfier = nn.Sequential()
        for i in range(6):
            self.classfier.add_module('classifier'+str(i), model_alexnet.classifier[i])
        self.out_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x

# print(network_dict)