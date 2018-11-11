import torch
import torch.nn as nn
import model

_alpha_keep = 0.9
_cross_stitch_unit = [[_alpha_keep, 1-_alpha_keep],
                      [1-_alpha_keep, _alpha_keep]]


class CrossStitchUnit(nn.Module):

    def __init__(self):
        super(CrossStitchUnit,self).__init__()
        self.cross_stitch_units = None

    def forward(self, input1, input2):
        assert input1.dim() == input2.dim() #share information only when input1 and input2 have the same shape
        output1, output2 = None, None
        if input1.dim() == 4: #n*c*w*h, output after conv net
            input_size = input1.size()
            input1 = input1.view(input1.size(0), input1.size(1), 1, -1)
            input2 = input2.view(input1.size(0), input1.size(1), 1, -1)
            input_total = torch.cat((input1, input2), dim=2)
            if self.cross_stitch_units is None:
                self.cross_stitch_units = torch.Tensor([_cross_stitch_unit for i in range(input1.size(1))])
                self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.cross_stitch_units, input_total)
            output1, output2 = torch.narrow(output_total, 2, 0, 1).view(input_size), \
                               torch.narrow(output_total, 2, 1, 1).view(input_size)
        elif input1.dim() == 2: #n*h, output after fc net
            input1 = input1.view(input1.size(0),  1, -1)
            input2 = input2.view(input1.size(0),  1, -1)
            input_total = torch.cat((input1, input2), dim=1)
            if self.cross_stitch_units is None:
                self.cross_stitch_units = torch.Tensor(_cross_stitch_unit)
                self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.cross_stitch_units, input_total)
            output1, output2 = torch.narrow(output_total, 1, 0, 1), torch.narrow(output_total, 1, 1, 1)
        return output1, output2


class CrossStitchNetwork(nn.Module):

    def __init__(self, source_architecture, target_architecture):
        super(CrossStitchNetwork,self).__init__()
        self.source_architecture = source_architecture
        self.target_architecture = target_architecture
        assert isinstance(self.source_architecture, (model.network_dict['AlexNetFc']))
        assert isinstance(self.target_architecture, (model.network_dict['AlexNetFc']))
        self.cross_stitch_units = [CrossStitchUnit() for i in range(5)]

    def forward(self, x):
        x1, x2 = x, x
        # consider about conv layer
        cross_unit_idx = 0
        for idx, m in enumerate(self.source_architecture.features.modules()):
            x1, x2 = m(x1), self.target_architecture.features[idx](x2)
            if isinstance(m, (nn.MaxPool2d)):
                x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                cross_unit_idx += 1
        for idx, m in enumerate(self.source_architecture.classifier.modules()):
            x1, x2 = m(x1), self.target_architecture.classifier[idx](x2)
            if isinstance(m, (nn.MaxPool2d)):
                x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                cross_unit_idx += 1
        return x1, x2





