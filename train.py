import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ISIC_2017_csvDataset(data.Dataset):
    def __init__(self, csv_root):
        self.csv_root = csv_root
        self.Haminfo = self.csv2tensors(self.csv_root)
        self.size = self.Haminfo[0].size(0)

    def __getitem__(self, index):
        return self.Haminfo[0][index]

    def csv2tensors(self, folder_path):
        # 读取指定文件夹中的所有csv文件并将数据存储为张量格式
        csv_tensors = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                # 读取csv文件
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                # 将读取的数据进行onehot编码
                le = LabelEncoder()
                # ISIC_2017
                data['age_approximate'] = le.fit_transform(data['age_approximate'])
                data['sex'] = le.fit_transform(data['sex'])
                # # ph2_Dataset
                # data['Histological Diagnosis'] = le.fit_transform(data['Histological Diagnosis'])
                # data['Common Nevus'] = le.fit_transform(data['Common Nevus'])
                # data['Atypical Nevus'] = le.fit_transform(data['Atypical Nevus'])
                # data['Melanoma'] = le.fit_transform(data['Melanoma'])
                # data['Asymmetry'] = le.fit_transform(data['Asymmetry'])
                # data['Pigment Network'] = le.fit_transform(data['Pigment Network'])
                # data['Dots/Globules'] = le.fit_transform(data['Dots/Globules'])
                # data['Streaks'] = le.fit_transform(data['Streaks'])
                # data['Regression Areas'] = le.fit_transform(data['Regression Areas'])
                # data['Blue-Whitish Veil'] = le.fit_transform(data['Blue-Whitish Veil'])
                # data['White'] = le.fit_transform(data['White'])
                # data['Red'] = le.fit_transform(data['Red'])
                # data['Light-Brown'] = le.fit_transform(data['Light-Brown'])
                # data['Dark-Brown'] = le.fit_transform(data['Dark-Brown'])
                # data['Blue-Gray'] = le.fit_transform(data['Blue-Gray'])
                # data['Black'] = le.fit_transform(data['Black'])
                # # HAM10000
                # data['dx'] = le.fit_transform(data['dx'])
                # data['dx_type'] = le.fit_transform(data['dx_type'])
                # data['age'] = le.fit_transform(data['age'])
                # data['sex'] = le.fit_transform(data['sex'])
                # data['localization'] = le.fit_transform(data['localization'])
                # print(data)
                # 读取存为numpy数组再转tensor
                temp_array = np.array(data)

                # 将数据转换为张量格式并添加到列表中
                tensor = torch.tensor(temp_array)
                tensor = tensor.float()
                # print(type(tensor))
                bn = nn.BatchNorm1d(2)  # 最后一个维度的大小。其实更加标准的说法是
                # 特征的数量，我们这里，一个数据有3个特征，一个特征就是一个数。所以用batchnorm1d。
                # 在计算机视觉中，一张图片就是一个数据，其特征数量就是其channel数量，就是这么定义的，别问为什么。
                # 此时一个特征就是一个224*224的矩阵，所以使用
                # batchnorm2d。
                tensor = bn(tensor)
                # print(tensor)
                csv_tensors.append(tensor)
        # 返回csv文件的张量列表
        return csv_tensors

    def __len__(self):
        return self.size

class ph2_csvDataset(data.Dataset):
    def __init__(self, csv_root):
        self.csv_root = csv_root
        self.Haminfo = self.csv2tensors(self.csv_root)
        self.size = self.Haminfo[0].size(0)

    def __getitem__(self, index):
        return self.Haminfo[0][index]

    def csv2tensors(self, folder_path):
        # 读取指定文件夹中的所有csv文件并将数据存储为张量格式
        csv_tensors = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                # 读取csv文件
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                # 将读取的数据进行onehot编码
                le = LabelEncoder()
                # ISIC_2017
                data['age_approximate'] = le.fit_transform(data['age_approximate'])
                data['sex'] = le.fit_transform(data['sex'])
                # ph2_Dataset
                data['Histological Diagnosis'] = le.fit_transform(data['Histological Diagnosis'])
                data['Common Nevus'] = le.fit_transform(data['Common Nevus'])
                data['Atypical Nevus'] = le.fit_transform(data['Atypical Nevus'])
                data['Melanoma'] = le.fit_transform(data['Melanoma'])
                data['Asymmetry'] = le.fit_transform(data['Asymmetry'])
                data['Pigment Network'] = le.fit_transform(data['Pigment Network'])
                data['Dots/Globules'] = le.fit_transform(data['Dots/Globules'])
                data['Streaks'] = le.fit_transform(data['Streaks'])
                data['Regression Areas'] = le.fit_transform(data['Regression Areas'])
                data['Blue-Whitish Veil'] = le.fit_transform(data['Blue-Whitish Veil'])
                data['White'] = le.fit_transform(data['White'])
                data['Red'] = le.fit_transform(data['Red'])
                data['Light-Brown'] = le.fit_transform(data['Light-Brown'])
                data['Dark-Brown'] = le.fit_transform(data['Dark-Brown'])
                data['Blue-Gray'] = le.fit_transform(data['Blue-Gray'])
                data['Black'] = le.fit_transform(data['Black'])
                # # HAM10000
                # data['dx'] = le.fit_transform(data['dx'])
                # data['dx_type'] = le.fit_transform(data['dx_type'])
                # data['age'] = le.fit_transform(data['age'])
                # data['sex'] = le.fit_transform(data['sex'])
                # data['localization'] = le.fit_transform(data['localization'])
                # print(data)
                # 读取存为numpy数组再转tensor
                temp_array = np.array(data)

                # 将数据转换为张量格式并添加到列表中
                tensor = torch.tensor(temp_array)
                tensor = tensor.float()
                # print(type(tensor))
                bn = nn.BatchNorm1d(16)  # 最后一个维度的大小。其实更加标准的说法是
                # 特征的数量，我们这里，一个数据有3个特征，一个特征就是一个数。所以用batchnorm1d。
                # 在计算机视觉中，一张图片就是一个数据，其特征数量就是其channel数量，就是这么定义的，别问为什么。
                # 此时一个特征就是一个224*224的矩阵，所以使用
                # batchnorm2d。
                tensor = bn(tensor)
                # print(tensor)
                csv_tensors.append(tensor)
        # 返回csv文件的张量列表
        return csv_tensors

    def __len__(self):
        return self.size

class Ham_csvDataset(data.Dataset):
    def __init__(self, csv_root):
        self.csv_root = csv_root
        self.Haminfo = self.csv2tensors(self.csv_root)
        self.size = self.Haminfo[0].size(0)

    def __getitem__(self, index):
        return self.Haminfo[0][index]

    def csv2tensors(self, folder_path):
        # 读取指定文件夹中的所有csv文件并将数据存储为张量格式
        csv_tensors = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                # 读取csv文件
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                # 将读取的数据进行onehot编码
                le = LabelEncoder()
                data['dx'] = le.fit_transform(data['dx'])
                data['dx_type'] = le.fit_transform(data['dx_type'])
                data['age'] = le.fit_transform(data['age'])
                data['sex'] = le.fit_transform(data['sex'])
                data['localization'] = le.fit_transform(data['localization'])
                # print(data)
                # 读取存为numpy数组再转tensor
                temp_array = np.array(data)

                # 将数据转换为张量格式并添加到列表中
                tensor = torch.tensor(temp_array)
                tensor = tensor.float()
                # print(type(tensor))
                bn = nn.BatchNorm1d(5)  # 最后一个维度的大小。其实更加标准的说法是
                # 特征的数量，我们这里，一个数据有3个特征，一个特征就是一个数。所以用batchnorm1d。
                # 在计算机视觉中，一张图片就是一个数据，其特征数量就是其channel数量，就是这么定义的，别问为什么。
                # 此时一个特征就是一个224*224的矩阵，所以使用
                # batchnorm2d。
                tensor = bn(tensor)
                # print(tensor)
                csv_tensors.append(tensor)
        # 返回csv文件的张量列表
        return csv_tensors

    def __len__(self):
        return self.size


def get_loader_csv(csv_root, batchsize, shuffle=True, pin_memory=True):
    dataset = Ham_csvDataset(csv_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory)
    return data_loader


class SkinDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = SkinDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('Snapshots/Res2net/res2net50.pth')
        model.load_state_dict(model_state)
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net101_v1b_26w_4s']))
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net152_v1b_26w_4s']))
    return model


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MLF(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(MLF, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class MLP(nn.Module):

    def __init__(self,
                 activation='relu',
                 dropout=0.1):
        super(MLP, self).__init__()
        self.input_dim = 5
        self.dimensions = [14, 32]
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.dimensions[0])])
        for din, dout in zip(self.dimensions[:-1], self.dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        x = x.float()
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears) - 1):
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


class MRML_Net(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, n_class=1,
                 mm_dim=1200,
                 factor=2,
                 activ_input='relu',
                 activ_output='relu',
                 normalize=True,
                 dropout_input=0.,
                 dropout_pre_norm=0.,
                 dropout_output=0.):
        super(MRML_Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Multilevel_fushion ----
        self.MLF = MLF(channel, n_class)
        # ----          MLP        ----
        self.MLP = MLP()
        # ----          MFB        ----
        self.input_dims0 = 32
        self.input_dims1 = 32 * 32
        self.input_dims2 = 44 * 44
        self.input_dims3 = 56 * 56
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dims1 = 32 * 32
        self.output_dims2 = 44 * 44
        self.output_dims3 = 56 * 56
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(self.input_dims0, mm_dim * factor)
        self.linear1 = nn.Linear(self.input_dims1, mm_dim * factor)
        self.linear2 = nn.Linear(self.input_dims2, mm_dim * factor)
        self.linear3 = nn.Linear(self.input_dims3, mm_dim * factor)
        self.linear_out1 = nn.Linear(mm_dim, self.output_dims1)
        self.linear_out2 = nn.Linear(mm_dim, self.output_dims2)
        self.linear_out3 = nn.Linear(mm_dim, self.output_dims3)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)


    def forward(self, x, y):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        # ---- high-level features ----
        # x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32


        # ---- Multilevel_fushion ----
        mlf = self.MLF(x4_rfb, x3_rfb, x2_rfb)
        y = self.MLP(y)

        lateral_map_5 = F.interpolate(mlf, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # mlf = mlf.flatten(1)
        # print(mlf.size())
        # ---- MFB Fushion Branch----
        fl_mlf = mlf.flatten(1)
        x0 = self.linear0(y)
        if mlf.size(2) == 32:
            x1 = self.linear1(fl_mlf)
        elif mlf.size(2) == 44:
            x1 = self.linear2(fl_mlf)
        else:
            x1 = self.linear3(fl_mlf)
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 * x1
        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)
        z = z.view(-1, int(z.size(1) / self.factor), self.factor)
        z = z.sum(2)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2, dim=0)

        if mlf.size(2) == 32:
            z = self.linear_out1(z)
        elif mlf.size(2) == 44:
            z = self.linear_out2(z)
        else:
            z = self.linear_out3(z)
        z = getattr(F, self.activ_output)(z)
        z = F.dropout(z, p=self.dropout_output, training=self.training)
        z = z.reshape([mlf.size(0), 1, mlf.size(2), mlf.size(3)])
        # print("z", z.size())
        # print(z)
        mlf = mlf + z

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(mlf, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32,
                                      mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

"""

Training 


"""


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

losslist = []

def train(train_loader, train_loader_csv, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    save_path = 'Snapshots/{}/'.format(opt.train_save)
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for pack, info in zip(enumerate(train_loader, start=1), enumerate(train_loader_csv, start=1)):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            i, package = pack
            j, haminfo = info
            images = package[0]
            gts = package[1]
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            haminfo = Variable(haminfo).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images, haminfo)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

    # 保存loss
    os.makedirs(save_path, exist_ok=True)
    losslist.append(loss_record2.show().cpu().detach().numpy())
    np.savetxt(save_path + 'train_loss.csv', losslist, delimiter=',')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'MFSNet_dx.pth')
        print('[Saving Snapshot:]', save_path + 'MFSNet_dx.pth')


# noinspection LanguageDetectionInspection
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=3, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.05, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=25, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='train/HAM10000', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='HAM10000/MFSNet_v2/testlabel')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = MRML_Net().cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    csv_root = '{}'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_loader_csv = get_loader_csv(csv_root, batchsize=opt.batchsize)
    total_step = len(train_loader)
    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, train_loader_csv, model, optimizer, epoch)
