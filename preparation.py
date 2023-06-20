# ---------------- native module  ----------------#
import os
import time
import numpy as np
import sys
from os.path import join as join
from os.path import exists as exists
import logging

# ---------------- torch module  ----------------#
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import timm

# ---------------- my module  ----------------#
from data.dataset import Feature_LIRIS, Original_LIRIS
from model.AmodelBase import *
from model.Ereproduce import *
from utils.utils import setup_seed, Marco_loss
from config import parser

args = parser.parse_args()


def dataset_preparation():
    train_names, val_names, test_names = dataset_split()
    mean, std = VA_calculate()

    train_dataset = Feature_LIRIS(train_names, VA=args.VA, length=args.length, samplingRate=args.samplingRate)
    val_dataset = Feature_LIRIS(val_names, VA=args.VA, length=args.length, samplingRate=1)
    test_dataset = Feature_LIRIS(test_names, VA=args.VA, length=args.length, samplingRate=1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    train_dict = {'loader': train_loader, 'mean': mean, 'std': std}
    val_dict = {'loader': val_loader, 'mean': mean, 'std': std}
    test_dict = {'loader': test_loader, 'mean': mean, 'std': std}

    Msg = ('train:{train}\n val:{val}\n test:{test}'.format(train=str(train_names), val=str(val_names), test=str(test_names)))
    logging.info(Msg)
    return train_dict, val_dict, test_dict


def experiment_preparation():
    setup_seed(args.random_seed)
    num_classes = numClass_calculate()
    modelConfig = modelConfig_choose(num_classes)
    model = model_choose(modelConfig)
    criterion = criterion_choose()
    optimizer = optimizer_choose(model)
    scheduler = lr_scheduler.StepLR(optimizer, 10, 0.5)

    Msg = ('total_params:{total_params}\n modelConfig:{modelConfig}\n'.format(total_params=sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000, modelConfig=str(modelConfig)))
    logging.info(Msg)
    return model, criterion, optimizer, scheduler


def logging_preparation():
    trainLog = pjoin('./log', args.model, args.VA, 'loss')
    if pexists(trainLog) is False:
        os.makedirs(trainLog)
    logging.basicConfig(filename='{}/{}_loss.log'.format(trainLog, args.logName), level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('------------------------------ start ------------------------------ \n' + '\t input argv:' + '\t'.join(sys.argv[1:]))
    output = ('device:{device}\t' 'model:{model}\t' 'criterion:{criterion}\t\n').format(device=args.cuda_devices, model=args.model, criterion=args.loss_type)
    logging.info(output)
    resultLogDir = pjoin('./log', args.model, args.VA, 'result', '{}_result'.format(args.logName))
    if pexists(resultLogDir) is False:
        os.makedirs(resultLogDir)
    return resultLogDir


# ------------------------------------------------------------------------------------------------
def dataset_split():
    all_video_dir = '/148Dataset/data-huang.maochun/datasets/LIRIS-ACCEDE/MEDIAEVAL-2018/afterProcess/frame'
    video_list = sorted(os.listdir(all_video_dir), key=lambda x: x)
    train_names = video_list[:44]
    # val_names = video_list[-10:]
    # test_names = video_list[44:56]
    val_names = video_list[44:56]
    test_names = video_list[-12:]

    if args.video_numbers != 0:
        train_names = train_names[:int(len(train_names) * args.video_numbers)]
        val_names = val_names[:int(len(val_names) * args.video_numbers)]
        test_names = val_names[:int(len(test_names) * args.video_numbers)]

    return train_names, val_names, test_names


def VA_calculate():
    if args.VA == 'VA':
        train_mean = [-0.02717169703556456, 0.11375035921999119]
        train_std = [0.3460118098861817, 0.29035571603148275]
    elif args.VA == 'V':
        train_mean = [-0.02717169703556456]
        train_std = [0.3460118098861817]
    elif args.VA == 'A':
        train_mean = [0.11375035921999119]
        train_std = [0.29035571603148275]
    return train_mean, train_std


def numClass_calculate():
    num_classes = 0
    if 'V' in args.VA:
        num_classes += 1
    if 'A' in args.VA:
        num_classes += 1
    return num_classes


def transfrom_get():
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

    transform_test = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform_train, transform_test


def optimizer_choose(model):
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    if args.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.9)
    return optimizer


def criterion_choose():
    if args.loss_type == 'mse':
        criterion = nn.MSELoss().cuda()
    elif args.loss_type == 'ccc':
        criterion = Marco_loss().cuda()
    return criterion


def modelConfig_choose(num_classes):
    modelConfig = dict()
    if args.modelConfig == '1':
        modelConfig['num_classes_frames'] = num_classes, args.length
        modelConfig['linear_temporal_fusion_dropout'] = 0.5, 0.5, 0.5
        modelConfig['dim_hiddenDim'] = 256, 64
        modelConfig['depth_heads'] = 1, 3
    if args.modelConfig == 'random':
        modelConfig['num_classes_frames'] = num_classes, args.length
        modelConfig['linear_temporal_fusion_dropout'] = random.choice([0.2, 0.5]), random.choice([0.2, 0.5]), random.choice([0.2, 0.5])
        modelConfig['dim_hiddenDim'] = random.choice([256, 64, 32, 16]), random.choice([64, 32, 16, 8])
        modelConfig['depth_heads'] = random.choice([1, 2, 3]), random.choice([1, 2, 3])
    return modelConfig


def model_choose(modelConfig):
    # ---------------- base model ---------------- #
    if args.model == 'baseModel_best':
        model = baseModel_best(modelConfig)

    # ---------------- exp1: fusion ---------------- #
    if args.model == 'baseModel_concat':
        model = baseModel_concat(1, frames=args.length)
    if args.model == 'baseModel_mean':
        model = baseModel_mean(1, frames=args.length)
    if args.model == 'baseModel_LFM':
        model = baseModel_LFM(1, frames=args.length)
    if args.model == 'baseModel_TFN':
        model = baseModel_TFN(1, frames=args.length)
    if args.model == 'baseModel_MFN':
        model = baseModel_MFN(1, frames=args.length)

    # ---------------- exp2: temporal ---------------- #
    if args.model == 'baseModel_lstm':
        model = baseModel_lstm(num_classes, frames=args.length)
    if args.model == 'baseModel_gru':
        model = baseModel_gru(num_classes, frames=args.length)
    if args.model == 'baseModel_nocontext':
        model = baseModel_nocontext(num_classes, frames=args.length)

    # ---------------- exp3: ablation ---------------- #
    model = nn.DataParallel(model).cuda()
    return model