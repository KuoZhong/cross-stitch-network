import torch
import torch.nn as nn
import model
import os

_alpha_keep = float(os.environ['ALPHA_SHARE'])
_cross_stitch_unit = [[1.0 - _alpha_keep, _alpha_keep],
                      ]


class ProgressUnit(nn.Module):

    def __init__(self, size):# size is the input size
        super(ProgressUnit,self).__init__()
        assert len(size)==4 or len(size)==2
        if len(size) == 4:
            self.progress_unit = nn.Parameter(torch.Tensor([_cross_stitch_unit for i in range(size[1])]))
            # self.cross_stitch_units.requires_grad_()
        elif len(size) == 2:
            self.progress_unit = nn.Parameter(torch.Tensor([_cross_stitch_unit for i in range(1)]))
            # self.cross_stitch_units.requires_grad_()

    def forward(self, input1, input2):
        assert input1.dim() == input2.dim() #share information only when input1 and input2 have the same shape
        output2 = None
        if input1.dim() == 4: #n*c*w*h, output after conv net
            input_size = input1.size()
            input1 = input1.view(input1.size(0), input1.size(1), 1, -1)
            input2 = input2.view(input1.size(0), input1.size(1), 1, -1)
            input_total = torch.cat((input1, input2), dim=2)
            # if self.cross_stitch_units is None:
            #     self.cross_stitch_units = torch.Tensor([_cross_stitch_unit for i in range(input1.size(1))])
            #     self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.progress_unit, input_total)
            output2 = output_total.view(input_size)
        elif input1.dim() == 2: #n*h, output after fc net
            input1 = input1.view(input1.size(0),  1, -1)
            input2 = input2.view(input1.size(0),  1, -1)
            input_total = torch.cat((input1, input2), dim=1)
            # if self.cross_stitch_units is None:
            #     self.cross_stitch_units = torch.Tensor(_cross_stitch_unit)
            #     self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.progress_unit, input_total)
            output2 = output_total.squeeze()
        return output2


class ProgressNetwork(nn.Module):

    def __init__(self, source_architecture, target_architecture):
        super(ProgressNetwork,self).__init__()
        self.source_architecture = source_architecture
        self.target_architecture = target_architecture
        assert isinstance(self.source_architecture, (model.network_dict['AlexNetFc']))
        assert isinstance(self.target_architecture, (model.network_dict['AlexNetFc']))
        self.progress_units = nn.ModuleList()
        size = [None, 3, None, None]
        for m in self.source_architecture.features.children():
            if isinstance(m, (nn.Conv2d)):
                size[1] = m.out_channels
            elif isinstance(m, (nn.Linear)):
                size = [None, 1]
            if isinstance(m, (nn.Linear, nn.MaxPool2d)):
                self.progress_units.append(ProgressUnit(size))
                print(type(m), self.progress_units)
        size = [None, 1]
        for m in self.source_architecture.classfier.children():
            if isinstance(m, (nn.Conv2d)):
                size[1] = m.out_channels
            elif isinstance(m, (nn.Linear)):
                size = [None, 1]
            if isinstance(m, (nn.Linear, nn.MaxPool2d)):
                self.progress_units.append(ProgressUnit(size))
                print(type(m), self.progress_units)

    def forward(self, x):
        x1, x2 = x, x
        # consider about conv layer
        progress_unit_idx = 0
        for (m1, m2) in zip(self.source_architecture.features.children(), self.target_architecture.features.children()):
            # print(type(m1),type(m2))
            # print(x1.shape, x2.shape)
            x1, x2 = m1(x1), m2(x2)
            if isinstance(m1, (nn.MaxPool2d, nn.Linear)):
                x2 = self.progress_units[progress_unit_idx](x1, x2)
                progress_unit_idx += 1
        x1, x2 = x1.contiguous().view(x1.size(0),-1), x2.contiguous().view(x2.size(0),-1)
        for (m1, m2) in zip(self.source_architecture.classfier.children(),self.target_architecture.classfier.children()):
            x1, x2 = m1(x1), m2(x2)
            if isinstance(m1, (nn.MaxPool2d, nn.Linear)):
                x2 = self.progress_units[progress_unit_idx](x1, x2)
                progress_unit_idx += 1
        return x1, x2