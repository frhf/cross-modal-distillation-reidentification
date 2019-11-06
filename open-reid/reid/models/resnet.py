from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet50_twostream', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0) 


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet50_twostream(**kwargs):
    
        
    #left_resnet50 = torchvision.models.resnet50(pretrained=True)
    #right_resnet50 = torchvision.models.resnet50(pretrained=True)

    #print(left_resnet50)
    #print(right_resnet50)


    #*super(ResNet, self).__init__()

#       self.depth = depth
#        self.pretrained = pretrained
#       self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
 #       if depth not in ResNet.__factory:
 #           raise KeyError("Unsupported depth:", depth)
 #       self.base = ResNet.__factory[depth](pretrained=pretrained)
    return ResNet_twostream(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


class ResNet_twostream(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet_twostream, self).__init__()
        
        self.depth = depth
        self.pretrained = pretrained

        self.left_base = ResNet_twostream.__factory[depth](pretrained=pretrained)
        self.right_base = ResNet_twostream.__factory[depth](pretrained=pretrained)

        self.left_out_plane = self.left_base.fc.in_features
        self.right_out_plane = self.right_base.fc.in_features

        #cat_out_planes = torch.cat((left_out_plane, right_out_plane), dim=0)

        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

             # Append new layers
        if self.has_embedding:
                self.feat = nn.Linear(self.left_out_plane+self.right_out_plane, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
        else:
                # Change the num_features to CNN output channels
                self.num_features = cat_out_planes
        if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
	# cheat to differentiate between the two streams (depth always 3 identical channels)

        if x[0][0][120][120]==x[0][1][120][120]==x[0][1][120][120] and x[0][0][120][60]==x[0][1][120][60]==x[0][1][120][60] and x[0][0][60][120]==x[0][1][60][120]==x[0][1][60][120]:
               for name, module in self.left_base._modules.items():
                  if name == 'avgpool':
                     break
                  x = module(x)
               #print("Depth")

               x = F.avg_pool2d(x, x.size()[2:])
               print(x.size)
               x = x.view(x.size(0), -1) 

        else:
               for name, module in self.right_base._modules.items():
                  if name == 'avgpool':
                     break
                  x = module(x)
               #print("RGB")

               x = F.avg_pool2d(x, x.size()[2:])
               print(x.size)
               x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0) 

