# ---------------- native module  ----------------#
import errno
import math
import os
import random
import shutil
import sys
import time
import warnings
from os.path import exists as pexists
from os.path import join as pjoin
import torch
import numpy as np
import six

from config import parser
from PIL import Image

from sklearn.metrics import mean_squared_error

args = parser.parse_args()


def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    v_pred = y_pred - pred_mean
    v_true = y_true - true_mean

    if np.sqrt(np.sum(v_pred**2)) == 0:
        v_pred = np.linspace(1e-6, 1e-6, v_pred.shape[0])
    if np.sqrt(np.sum(v_true**2)) == 0:
        v_true = np.linspace(1e-6, 1e-6, v_true.shape[0])
    assert np.sqrt(np.sum(v_pred**2)) != 0
    assert np.sqrt(np.sum(v_true**2)) != 0
    rho = np.sum(v_pred * v_true) / (np.sqrt(np.sum(v_pred**2)) * np.sqrt(np.sum(v_true**2)))
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (std_predictions**2 + std_gt**2 + (pred_mean - true_mean)**2)
    return ccc


def pcc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    v_pred = y_pred - pred_mean
    v_true = y_true - true_mean

    if np.sqrt(np.sum(v_pred**2)) == 0:
        v_pred = np.linspace(1e-6, 1e-6, v_pred.shape[0])
    if np.sqrt(np.sum(v_true**2)) == 0:
        v_true = np.linspace(1e-6, 1e-6, v_true.shape[0])
    assert np.sqrt(np.sum(v_pred**2)) != 0
    assert np.sqrt(np.sum(v_true**2)) != 0
    rho = np.sum(v_pred * v_true) / (np.sqrt(np.sum(v_pred**2)) * np.sqrt(np.sum(v_true**2)))
    return rho


class Marco_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def zero_judge(self, x, y):
        # 传引用
        smallNum = torch.zeros_like(x) + 1e-5
        x = torch.where(x == 0, smallNum, x)
        y = torch.where(y == 0, smallNum, y)
        return x, y

    def forward(self, x, y):
        cccs = 0
        # 取VA值
        for i in range(x.size(-1)):
            if len(x.shape) == 3:
                x_i = x[:, :, i].view(-1)
                y_i = y[:, :, i].view(-1)
            if len(x.shape) == 2:
                x_i = x[:, i]
                y_i = y[:, i]

            x_m, x_s = torch.mean(x_i), torch.std(x_i)
            y_m, y_s = torch.mean(y_i), torch.std(y_i)

            vx = x_i - torch.mean(x_i)
            vy = y_i - torch.mean(y_i)
            vx, vy = self.zero_judge(vx, vy)

            rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
            ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
            cccs += ccc
        return 2 - cccs


class CCC_Calculator():
    def __init__(self, train_mean=[0, 0], train_std=[1, 1]):
        self.targets_list, self.predicts_list = [], []
        self.train_mean = train_mean
        self.train_std = train_std
        self.result = ''
        self.length = args.length
        self.samplingRate = args.samplingRate
        self.slope = 0.05

    def marco_correct(self, pred_val_y, train_mean, train_std):
        pred_correct = np.zeros(pred_val_y.shape)
        for i in range(pred_val_y.shape[-1]):
            try:
                val_std = np.std(pred_val_y[:, i])
                mean = np.mean(pred_val_y[:, i])
            except:
                val_std = torch.std(pred_val_y[:, i])
                mean = torch.mean(pred_val_y[:, i])
            pred_correct[:, i] = train_mean[i] + (pred_val_y[:, i] - mean) * train_std[i] / val_std
        return pred_correct

    def marco_correct2(self, pred_val_y):
        length, slope = self.length // self.samplingRate, self.slope
        total_num = pred_val_y.shape[0] // length
        assert pred_val_y.shape[0] % length == 0
        pred_correct = np.zeros(pred_val_y.shape)
        i = 0
        while i < total_num:
            seq = pred_val_y[i * length:(i + 1) * length]
            y_mean = (seq[length // 2 - 1] + seq[length // 2]) / 2
            seq = np.insert(seq, length // 2, y_mean, axis=0)
            for j in range(pred_val_y.shape[1]):
                V_OR_A_seq = seq[:, j]
                seq_right = [
                    V_OR_A_seq[i + 1] - slope if V_OR_A_seq[i] - V_OR_A_seq[i + 1] < -slope else V_OR_A_seq[i + 1] + slope if V_OR_A_seq[i] - V_OR_A_seq[i + 1] > slope else V_OR_A_seq[i]
                    for i in range(length // 2, length)
                ]
                seq_left = [
                    V_OR_A_seq[i - 1] - slope if V_OR_A_seq[i] - V_OR_A_seq[i - 1] < -slope else V_OR_A_seq[i - 1] + slope if V_OR_A_seq[i] - V_OR_A_seq[i - 1] > slope else V_OR_A_seq[i]
                    for i in range(length // 2, 0, -1)
                ]
                pred_correct[i * length:(i + 1) * length, j] = np.array(seq_left + seq_right)
            i += 1
        return pred_correct

    def update(self, real, pred):
        self.targets_list.append(real)
        self.predicts_list.append(pred)

    def sumUP(self):
        self.targets = np.concatenate([array for array in self.targets_list], axis=0)
        self.targets = self.targets.reshape(-1, self.targets.shape[-1])
        self.predicts = np.concatenate([array for array in self.predicts_list], axis=0)
        self.predicts = self.predicts.reshape(-1, self.predicts.shape[-1])
        self.correct_predicts = self.marco_correct(self.predicts, self.train_mean, self.train_std)
        self.correct_predicts2 = self.marco_correct2(self.predicts)

    def reset_result(self):
        self.result = ''

    def calculate_ccc(self):
        self.ccc_score = [ccc(self.targets[:, i], self.predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.ccc_corr = [ccc(self.targets[:, i], self.correct_predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.ccc_corr2 = [ccc(self.targets[:, i], self.correct_predicts2[:, i]) for i in range(self.targets.shape[-1])]

        self.pcc_score = [pcc(self.targets[:, i], self.predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.pcc_corr = [pcc(self.targets[:, i], self.correct_predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.pcc_corr2 = [pcc(self.targets[:, i], self.correct_predicts2[:, i]) for i in range(self.targets.shape[-1])]

        tmp = ['ccc_{}: {:.4f}({:.4f}/{:.4f})\t'.format(i, self.ccc_score[i], self.ccc_corr[i], self.ccc_corr2[i]) for i in range(self.targets.shape[-1])]
        tmp += ['pcc_{}: {:.4f}({:.4f}/{:.4f})\t'.format(i, self.pcc_score[i], self.pcc_corr[i], self.pcc_corr2[i]) for i in range(self.targets.shape[-1])]
        for i in range(len(tmp)):
            self.result = self.result + tmp[i]

    def calculate_mse(self):
        mse_loss = [mean_squared_error(self.targets[:, i], self.predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.mse_corr = [mean_squared_error(self.targets[:, i], self.correct_predicts[:, i]) for i in range(self.targets.shape[-1])]
        self.mse_corr2 = [mean_squared_error(self.targets[:, i], self.correct_predicts2[:, i]) for i in range(self.targets.shape[-1])]
        tmp = ['mse_{}: {:.4f}({:.4f}/{:.4f})\t'.format(i, mse_loss[i], self.mse_corr[i], self.mse_corr2[i]) for i in range(self.targets.shape[-1])]
        for i in range(len(tmp)):
            self.result = self.result + tmp[i]

    def process(self):
        self.sumUP()
        self.reset_result()
        self.calculate_ccc()
        self.calculate_mse()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Judger():
    def __init__(self, early_stop=10, best_loss=1000):
        self.best_loss = best_loss
        self.early_stop = early_stop
        self.not_become_better = 0

    def update(self, new_loss):
        self.new_loss = new_loss

    def saveJudge(self, model, epoch):
        if self.new_loss < self.best_loss:
            self.best_loss = self.new_loss
            # self.save(model, 'best')
        else:
            self.not_become_better += 1
            # self.save(model, 'regular_save')

    def save(self, model, event='regular_save'):
        if pexists(pjoin(args.save_root, args.model)) is False:
            os.mkdir(pjoin(args.save_root, args.model))
        state = {'net': model.state_dict(), 'loss': self.new_loss, 'event': event}
        if event == 'regular_save':
            torch.save(state, '%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.model, args.logName))
            with open('%s/%s_notes.txt' % (args.save_root, args.model), 'a') as writer:
                writer.write('checkpoint, los:{}\n'.format(self.new_loss))
        if event == 'best':
            torch.save(state, '%s/%s/%s_bestLoss.pth.tar' % (args.save_root, args.model, args.logName))
            with open('%s/%s_notes.txt' % (args.save_root, args.model), 'a') as writer:
                writer.write('best, los:{}\n'.format(self.new_loss))

    def isStop(self):
        if self.not_become_better > self.early_stop:
            return True
        else:
            return False


class AverageMeter():
    '''
        computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    term_width = 5
    TOTAL_BAR_LENGTH = 65.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def msgWriteDown(videoPath, clipIndex, label, predicted, savePath):
    clipIndex, label, predicted = clipIndex.data.cpu().numpy(), label.data.cpu().numpy(), predicted.data.cpu().numpy()
    result = [i for i in zip(videoPath, clipIndex, label, predicted)]
    with open(savePath, 'a') as f:
        for inf in result:
            for msg in inf:
                f.write(str(msg))
                f.write(',')
            f.write('\n')


def save_to_file(file_name, contents):
    fh = open(file_name, 'a')
    fh.write(contents)
    fh.close()
