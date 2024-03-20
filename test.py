import torch
import numpy as np
import os, argparse
from scipy import misc
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import pandas as pd
import imageio
import torch.utils.data as data
import skimage.io as io
from skimage import img_as_ubyte
import torch
from sklearn.preprocessing import LabelEncoder
import cv2
from torch.autograd import Variable
from datetime import datetime
import os
from PIL import Image
import torchvision.transforms as transforms
import math
import torch.nn as nn
import torch.nn.functional as F
from train import MFSNet
# from thop import profile
# from thop import clever_format





class Dataset(data.Dataset):

    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if
                    f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(edge_root) != 0:
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if
                          f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG')]
            self.edges = sorted(self.edges)
        else:
            self.edge_flage = False

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
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


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = Dataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset_csv:
    def __init__(self, csv_root):
        self.csv_root = csv_root
        self.Haminfo = self.csv2tensors(self.csv_root)
        self.size = self.Haminfo[0].size(0)
        self.index = -1

    def load_data(self):
        self.index += 1
        # print(self.Haminfo[0][1226])
        # print(self.Haminfo[0][1174])
        # print(self.Haminfo[0][479])
        print(self.Haminfo[0][227])
        # print(self.Haminfo[0][1263])
        return self.Haminfo[0][self.index]

    def csv2tensors(self, folder_path):
        csv_tensors = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                le = LabelEncoder()
                # data['age_approximate'] = le.fit_transform(data['age_approximate'])
                # data['sex'] = le.fit_transform(data['sex'])
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
                data['dx'] = le.fit_transform(data['dx'])
                data['dx_type'] = le.fit_transform(data['dx_type'])
                data['age'] = le.fit_transform(data['age'])
                data['sex'] = le.fit_transform(data['sex'])
                data['localization'] = le.fit_transform(data['localization'])
                # print(data)
                temp_array = np.array(data)

                tensor = torch.tensor(temp_array)
                tensor = tensor.float()
                # print(type(tensor))
                bn = nn.BatchNorm1d(5) 
                # batchnorm2d。
                tensor = bn(tensor)
                # print(tensor)
                csv_tensors.append(tensor)
        return csv_tensors


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        image = self.transform(image).unsqueeze(0)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '_segmentation.png'
        if name.endswith('.bmp'):
            name = name.split('.bmp')[0] + '_lesion.bmp'

        self.index += 1

        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--model_path', type=str, default='./Snapshots/HAM10000/MFSNet_local.pth')
    parser.add_argument('--data_path', type=str, default='test/HAM10000', help='Directory of test images')
    parser.add_argument('--save_path', type=str, default='test/HAM10000/outputs/',
                        help='Directory where prediction masks will be saved.')

    opt = parser.parse_args()
    data_path = opt.data_path
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = MFSNet()
    model.load_state_dict(torch.load(opt.model_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    # gt_root = '{}/masks/'.format(data_path)
    csv_root = '{}/csv'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    test_loader_csv = test_dataset_csv(csv_root)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        haminfo = test_loader_csv.load_data()

        image = image.cuda()
        haminfo = haminfo.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(image, haminfo)

        res = lateral_map_2
        # print("res", res.size())
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # lateral_edge = lateral_edge.data.cpu().numpy().squeeze()
        inv_map = lateral_map_4.max() - lateral_map_4

        inv_map = inv_map.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_4 = lateral_map_4.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_3 = lateral_map_3.data.cpu().numpy().squeeze()
        lateral_map_5 = lateral_map_5.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print("res", res)
        # print(img_as_ubyte(res))
        io.imsave(save_path + name, img_as_ubyte(res))
