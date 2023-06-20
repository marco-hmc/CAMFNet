from torchvision import transforms
from torch.autograd import Variable as Variable
from torch.multiprocessing import Pool, Process, set_start_method
from torchvision.utils import save_image
import torch.utils.data as data

# --------------- native module --------------- #
import os
from os.path import join as pjoin
from os.path import exists as pexists
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
import sys

# --------------- my module --------------- #

from utils.utils import *
from config import parser

args = parser.parse_args()


class Feature_LIRIS(data.Dataset):
    def __init__(self, movie_list, length=16, VA='V', samplingRate=1):
        self.movie_list = movie_list
        self.length = length
        self.VA = VA
        assert VA == 'V' or VA == 'A'
        self.samplingRate = samplingRate
        assert self.samplingRate % 2 == 0 or self.samplingRate == 1
        self.annotDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/annotations'
        self.imagesDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_images_' + str(self.length)
        self.audiosDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_audios_' + str(self.length)
        self.vggishDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_vggish_' + str(self.length)
        self.scenesDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_scenes_' + str(self.length)
        self.faceDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_face_' + str(self.length)
        self.videoDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_newVideo_' + str(self.length)
        self.prevDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/previous_' + str(self.length) + '/' + self.VA
        # self.videoDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_video_' + str(self.length)
        # self.opticalDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_optical_' + str(self.length)
        self.order_preparation()

    def order_preparation(self):
        self.movie_order_list = []
        self.total_length = 0
        for movie in self.movie_list:
            movie_path = pjoin(self.audiosDir, movie)
            self.total_length += len(os.listdir(movie_path))
            self.movie_order_list.append([movie, self.total_length, len(os.listdir(movie_path))])
        # 因爲idx2moviePointer取第一個的時候，要減去movie_order_list[-1]
        self.movie_order_list.append(['boarden_situation', 0, 0])

    def idx2moviePointer(self, idx):
        for i in range(len(self.movie_order_list)):
            movie_name = self.movie_order_list[i][0]
            total_length = self.movie_order_list[i][1]
            movie_length = self.movie_order_list[i][2]
            if idx < total_length:
                correct_idx = idx - self.movie_order_list[i - 1][1]
                return movie_name, correct_idx

    def feature_load(self, path, movie, idx):
        dataPath = pjoin(path, movie, '{:05d}-{:d}.npy'.format(idx, self.length))
        data = np.load(dataPath)
        if path == self.videoDir:
            return data
        else:
            data = np.stack([data[i] for i in np.arange(0, self.length, self.samplingRate)], 0)
            return data

    def va_read(self, idx):
        with open(self.annot_file, 'r') as reader:
            annot_lines = reader.readlines()
            annot_lines.pop(0)

            label_list = [tmp.rstrip().split('\t') for tmp in annot_lines[idx * self.length:(idx + 1) * self.length]]
            valence_mean = [float(label_list[i][1]) for i in np.arange(0, len(label_list), self.samplingRate)]
            arousal_mean = [float(label_list[i][2]) for i in np.arange(0, len(label_list), self.samplingRate)]
            if self.VA == 'VA':
                va_label = np.array([valence_mean, arousal_mean])
            elif self.VA == 'V':
                va_label = np.array([valence_mean])
            elif self.VA == 'A':
                va_label = np.array([arousal_mean])
        return torch.from_numpy(va_label).t()

    def __getitem__(self, index):
        movie_name, correct_idx = self.idx2moviePointer(index)
        images_data = self.feature_load(self.imagesDir, movie_name, correct_idx)
        audios_data = self.feature_load(self.audiosDir, movie_name, correct_idx)
        vggish_data = self.feature_load(self.vggishDir, movie_name, correct_idx)
        scenes_data = self.feature_load(self.scenesDir, movie_name, correct_idx)
        face_data = self.feature_load(self.faceDir, movie_name, correct_idx)
        video_data = self.feature_load(self.videoDir, movie_name, correct_idx)
        prev_data = self.feature_load(self.prevDir, movie_name, correct_idx)

        self.annot_file = pjoin(self.annotDir, '{}_Valence-Arousal.txt'.format(movie_name))
        labels = self.va_read(int(correct_idx))

        return images_data, audios_data, vggish_data, scenes_data, face_data, video_data, labels, movie_name, correct_idx, prev_data

    def __len__(self):
        return self.total_length


def test_Feature_LIRIS():
    base_dir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/frame'
    train_names = sorted(os.listdir(base_dir), key=lambda x: str(x.split('.')[0]))[:-1]

    train_dataset = Feature_LIRIS(train_names, length=64, samplingRate=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0, pin_memory=False, shuffle=False)
    for i, data_batch in enumerate(train_loader):
        images_data, audios_data, vggish_data, scenes_data, face_data, video_data, labels, movie_name, correct_idx, prev_labels = data_batch
        pass


class Original_LIRIS(data.Dataset):
    def __init__(self, movie_list, transform=None, length=16, VA='VA', samplingRate=1):
        self.movie_list = movie_list
        self.length = length
        self.VA = VA
        self.samplingRate = samplingRate
        assert self.samplingRate % 2 == 0 or self.samplingRate == 1
        self.transform = transform
        self.annotDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/annotations'
        self.imagesDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/frame'
        self.audiosDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_audios_' + str(self.length)
        # self.scenesDir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/features/features_scenes_' + str(self.length)
        self.order_preparation()

    def order_preparation(self):
        self.movie_order_list = []
        self.total_length = 0
        for movie in self.movie_list:
            movie_path = pjoin(self.audiosDir, movie)
            self.total_length += len(os.listdir(movie_path))
            self.movie_order_list.append([movie, self.total_length, len(os.listdir(movie_path))])
        # 因爲idx2moviePointer取第一個的時候，要減去movie_order_list[-1]
        self.movie_order_list.append(['boarden_situation', 0, 0])

    def idx2moviePointer(self, idx):
        for i in range(len(self.movie_order_list)):
            movie_name = self.movie_order_list[i][0]
            total_length = self.movie_order_list[i][1]
            movie_length = self.movie_order_list[i][2]
            if idx < total_length:
                correct_idx = idx - self.movie_order_list[i - 1][1]
                return movie_name, correct_idx

    def feature_load(self, path, movie, idx):
        dataPath = pjoin(path, movie, '{:05d}-{:d}.npy'.format(idx, self.length))
        data = np.load(dataPath)
        data = np.stack([data[i] for i in np.arange(0, self.length, self.samplingRate)], 0)
        return data

    def va_read(self, idx):
        with open(self.annot_file, 'r') as reader:
            annot_lines = reader.readlines()
            annot_lines.pop(0)

            label_list = [tmp.rstrip().split('\t') for tmp in annot_lines[idx * self.length:(idx + 1) * self.length]]
            # valence_mean = [float(label_list[i][1]) for i in range(len(label_list))]
            # arousal_mean = [float(label_list[i][2]) for i in range(len(label_list))]
            valence_mean = [float(label_list[i][1]) for i in np.arange(0, len(label_list), self.samplingRate)]
            arousal_mean = [float(label_list[i][2]) for i in np.arange(0, len(label_list), self.samplingRate)]
            if self.VA == 'VA':
                va_label = np.array([valence_mean, arousal_mean])
            elif self.VA == 'V':
                va_label = np.array([valence_mean])
            elif self.VA == 'A':
                va_label = np.array([arousal_mean])
        return torch.from_numpy(va_label).t()

    def imgs_load(self, path, movie, idx):
        # imgs start from 1 not 0
        imgs_list = [pjoin(path, movie, '{:05d}.png'.format(i + 1)) for i in np.arange(idx * self.length, (idx + 1) * self.length, self.samplingRate)]
        img_data_list = []
        for img_path in imgs_list:
            pil_img = Image.open(img_path)
            if pil_img.mode == "L":
                pil_img = pil_img.convert("RGB")
            if self.transform:
                img_data_list.append(self.transform(pil_img).unsqueeze(0))
            else:
                # TODO: 原始數據的忽略了
                raise ValueError("transfrom 必須有")
                img_data_list.append(pil_img)
        return torch.cat([img for img in img_data_list], axis=0)

    def __getitem__(self, index):
        movie_name, correct_idx = self.idx2moviePointer(index)
        images_data = self.imgs_load(self.imagesDir, movie_name, correct_idx)
        audios_data = self.feature_load(self.audiosDir, movie_name, correct_idx)
        # scenes_data = self.feature_load(self.scenesDir, movie_name, correct_idx)
        self.annot_file = pjoin(self.annotDir, '{}_Valence-Arousal.txt'.format(movie_name))
        labels = self.va_read(int(correct_idx))

        prevIndex = 0 if index == 0 else index - 1
        movie_name, correct_idx = self.idx2moviePointer(prevIndex)
        self.annot_file = pjoin(self.annotDir, '{}_Valence-Arousal.txt'.format(movie_name))
        prev_labels = self.va_read(int(correct_idx))
        return images_data, audios_data, labels, movie_name, correct_idx, prev_labels

    def __len__(self):
        return self.total_length


def test_Original_LIRIS():
    base_dir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/frame'
    train_names = sorted(os.listdir(base_dir), key=lambda x: str(x.split('.')[0]))[:]

    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(448),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = Original_LIRIS(train_names, transform=transform_train, samplingRate=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0, pin_memory=True, shuffle=False)
    for i, data_batch in enumerate(train_loader):
        images_data, audios_data, labels, movie_name, correct_idx, prev_labels = data_batch
        pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # test_Original_LIRIS()
    test_Feature_LIRIS()
    print('done')