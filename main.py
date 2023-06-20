import os
from os.path import join as join
from os.path import exists as exists
import sys

import torch
import time
import logging
import numpy as np
from preparation import experiment_preparation, dataset_preparation, logging_preparation
from utils.utils import AverageMeter, CCC_Calculator, Judger, save_to_file
from config import parser

args = parser.parser_args()


def train(train_dict, model, criterion, optimizer, epoch, resultLog):
    dataloader = train_dict['loader']
    my_ccc_calculator = CCC_Calculator(train_dict['mean'], train_dict['std'])
    batch_time, data_time, losses, end = AverageMeter(), AverageMeter(), AverageMeter(), time.time()

    optimizer.zero_grad()
    model.train()
    for i, data_batch in enumerate(dataloader):
        data_time.update(time.time() - end)

        img_data, audio_data, vggish_data, scene_data, face_data, video_data, va_label, movie_name, idxCorrect, prev_data = data_batch
        img_data = img_data.type('torch.FloatTensor').cuda()
        audio_data = audio_data.type('torch.FloatTensor').cuda()
        vggish_data = vggish_data.type('torch.FloatTensor').cuda()
        scene_data = scene_data.type('torch.FloatTensor').cuda()
        face_data = face_data.type('torch.FloatTensor').cuda()
        video_data = video_data.type('torch.FloatTensor').cuda()
        va_label = va_label.type('torch.FloatTensor').cuda()
        prev_data = prev_data.type('torch.FloatTensor').cuda()

        optimizer.zero_grad()
        output = model(img_data, audio_data, vggish_data, scene_data, face_data, video_data, prev_data)
        loss = criterion(output, va_label)
        loss.backward()
        optimizer.step()

        my_ccc_calculator.update(real=va_label.data.cpu().numpy(), pred=output.data.cpu().numpy())
        batch_time.update(time.time() - end)
        losses.update(loss.item(), img_data.size(0))
        end = time.time()

        if i % (len(dataloader) // args.printFreq) == 0 and i != 0:
            outputInfo = ('Epoch: [{0}][clip:{1}/{2}\t -lr: {lr:.6f}] {3} \n'
                          '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          '\tData {data_time.val:.3f} ({data_time.avg:.3f})\n'
                          '\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(dataloader),
                                                                          movie_name,
                                                                          lr=optimizer.param_groups[-1]['lr'],
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss=losses))
            my_ccc_calculator.process()
            outputInfo = outputInfo + '\t' + my_ccc_calculator.result + '\n'
            logging.info(outputInfo)
    my_ccc_calculator.process()
    result = np.concatenate((my_ccc_calculator.targets, my_ccc_calculator.predicts), 1)
    with open(resultLog, 'w') as f:
        np.savetxt(resultLog, result, delimiter=',')


@torch.no_grad()
def validate(val_dict, model, criterion, epoch, resultLog):
    dataloader = val_dict['loader']
    my_ccc_calculator = CCC_Calculator(val_dict['mean'], val_dict['std'])
    losses = AverageMeter()

    model.eval()
    for i, data_batch in enumerate(dataloader):
        img_data, audio_data, vggish_data, scene_data, face_data, video_data, va_label, movie_name, idxCorrect, prev_data = data_batch
        img_data = img_data.type('torch.FloatTensor').cuda()
        audio_data = audio_data.type('torch.FloatTensor').cuda()
        vggish_data = vggish_data.type('torch.FloatTensor').cuda()
        scene_data = scene_data.type('torch.FloatTensor').cuda()
        face_data = face_data.type('torch.FloatTensor').cuda()
        video_data = video_data.type('torch.FloatTensor').cuda()
        va_label = va_label.type('torch.FloatTensor').cuda()
        prev_data = prev_data.type('torch.FloatTensor').cuda()

        output = model(img_data, audio_data, vggish_data, scene_data, face_data, video_data, prev_data)
        loss = criterion(output, va_label)
        my_ccc_calculator.update(real=va_label.data.cpu().numpy(), pred=output.data.cpu().numpy())

        losses.update(loss.item(), img_data.size(0))

        if i % (len(dataloader) // args.printFreq) == 0 and i != 0:
            my_ccc_calculator.process()
            outputInfo = ' '.join(['Validation : [{0}][{1}]\t Loss:{loss.val:.4f}({loss.avg:.4f})'.format(i, len(dataloader), loss=losses)])
            outputInfo = outputInfo + '\t' + my_ccc_calculator.result + '\n'
            logging.info(outputInfo)

    my_ccc_calculator.process()
    outputMsg = ' '.join(['Epoch: [{0}] ---VAL--- Loss:{loss.val:.4f}({loss.avg:.4f})'.format(epoch, loss=losses)]) + '\t' + my_ccc_calculator.result + '\n'
    logging.info(outputMsg)
    result = np.concatenate((my_ccc_calculator.targets, my_ccc_calculator.predicts), 1)
    with open(resultLog, 'w') as f:
        np.savetxt(resultLog, result, delimiter=',')
    return losses.avg, my_ccc_calculator.result, result


def test(test_dict, model, criterion, resultLog):
    dataloader = test_dict['loader']
    my_ccc_calculator = CCC_Calculator(test_dict['mean'], test_dict['std'])
    losses = AverageMeter()

    model.eval()
    for i, data_batch in enumerate(dataloader):
        img_data, audio_data, vggish_data, scene_data, face_data, video_data, va_label, movie_name, idxCorrect, prev_data = data_batch
        img_data = img_data.type('torch.FloatTensor').cuda()
        audio_data = audio_data.type('torch.FloatTensor').cuda()
        vggish_data = vggish_data.type('torch.FloatTensor').cuda()
        scene_data = scene_data.type('torch.FloatTensor').cuda()
        face_data = face_data.type('torch.FloatTensor').cuda()
        video_data = video_data.type('torch.FloatTensor').cuda()
        va_label = va_label.type('torch.FloatTensor').cuda()
        prev_data = prev_data.type('torch.FloatTensor').cuda()

        output = model(img_data, audio_data, vggish_data, scene_data, face_data, video_data, prev_data)
        loss = criterion(output, va_label)
        my_ccc_calculator.update(real=va_label.data.cpu().numpy(), pred=output.data.cpu().numpy())

        losses.update(loss.item(), img_data.size(0))

        if i % (len(dataloader) // args.printFreq) == 0 and i != 0:
            my_ccc_calculator.process()
            outputInfo = ' '.join(['Test : [{0}][{1}]\t Loss:{loss.val:.4f}({loss.avg:.4f})'.format(i, len(dataloader), loss=losses)])
            outputInfo = outputInfo + '\t' + my_ccc_calculator.result + '\n'
            logging.info(outputInfo)

    my_ccc_calculator.process()
    outputMsg = ' '.join(['Epoch: [{0}] ---TEST--- Loss:{loss.val:.4f}({loss.avg:.4f})'.format(epoch, loss=losses)]) + '\t' + my_ccc_calculator.result + '\n'
    logging.info(outputMsg)
    result = np.concatenate((my_ccc_calculator.targets, my_ccc_calculator.predicts), 1)
    with open(resultLog, 'w') as f:
        np.savetxt(resultLog, result, delimiter=',')
    return losses.avg, my_ccc_calculator.result, result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    model, criterion, optimizer, scheduler = experiment_preparation()
    train_dict, val_dict, test_dict = dataset_preparation()
    result_dir = logging_preparation()
    myJudger = Judger(early_stop=10)

    for epoch in range(args.epochs):
        train_predict_analysis = train(train_dict, model, criterion, optimizer, scheduler, epoch, join(result_dir, 'train.csv'))
        val_loss, val_predict_analysis, val_predict_data = validate(val_dict, model, criterion, optimizer, epoch, join(result_dir, 'val.csv'))
        scheduler.stop()

        myJudger.update(val_loss, val_predict_analysis, val_predict_data)
        # myJudger.saveJudger(model, epoch)
        if myJudger.isStop() is True:
            break
    test_loss, test_predict_analysis, test_predict_data = test(test_dict, model, criterion, optimizer, join(result_dir, 'test_csv'))

    logMsg = 'input argv: \t' + ''.join(sys.argv[1:]) + '\n'
    logMsg = logMsg + '\t\t  train reslt is:' + train_predict_analysis + '\n'
    logMsg = logMsg + '\t\t  val reslt is:' + myJudger.val_predict_analysis + '\n'
    logMsg = logMsg + '\t\t  test reslt is:' + test_predict_analysis + '\n'
    logging.info(logMsg)
    save_to_file('./log/all.txt', logMsg)

    bestResultLog = join(result_dir, 'best_val.csv')
    with open(bestResultLog, 'w') as f:
        np.savetxt(bestResultLog, myJudger.val_predict_data, delimiter=',')
